import logging
from typing import Callable, List, Union

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.infer.autoguide.guides import deep_getattr, deep_setattr
from pyro.nn import PyroModule, PyroParam
from pyro.optim import Adam, ClippedAdam
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MuVI(PyroModule):
    def __init__(
        self,
        n_factors: int,
        observations: List[Union[np.ndarray, pd.DataFrame]],
        view_names: List[str] = None,
        covariates: Union[np.ndarray, pd.DataFrame] = None,
        likelihoods: List[str] = None,
        use_gpu: bool = True,
        **kwargs,
    ):
        """MuVI module.

        Parameters
        ----------
        n_factors : int
            Number of latent factors
        observations : List[Union[np.ndarray, pd.DataFrame]]
            List of M N x D matrices of observations
        view_names : List[str]
            List of names for each view
        covariates : Union[np.ndarray, pd.DataFrame]
            N x P matrix of covariates
        likelihoods : List[str], optional
            List of likelihoods for each view,
            either 'normal' or 'bernoulli', by default None
        use_gpu : bool, optional
            Whether to use a GPU, by default True

        Raises
        ------
        ValueError
            Missing or negative number of factors
        ValueError
            Missing observations
        """
        super().__init__(name="MuVI")

        if n_factors is None or n_factors <= 0:
            raise ValueError("Invalid `n_factors`, please pass a positive integer.")
        self.n_factors = n_factors

        if observations is None:
            raise ValueError(
                "Observations is None, please pass a valid list of observations."
            )
        if not isinstance(observations, list):
            observations = [observations]
        if len(observations) == 1:
            logger.warning("Running MuVI on a single view.")

        self.observations = self._setup_observations(observations, view_names)
        self.likelihoods = self._setup_likelihoods(likelihoods)
        self.covariates = self._setup_covariates(covariates)

        self.device = torch.device("cpu")
        if use_gpu and torch.cuda.is_available():
            logger.info("GPU available, running all computations on the GPU.")
            self.device = torch.device("cuda")
        self.to(self.device)

        self.kwargs = kwargs

        self.factor_names = None
        self.prior_masks = None
        self.prior_scales = None
        self.model = None
        self.guide = None
        self._is_built = False

    def _setup_observations(self, observations, view_names):

        n_samples = observations[0].shape[0]
        sample_names = None
        n_features = []
        feature_offsets = [0]
        feature_names = []
        numpy_observations = []
        for m, y in enumerate(observations):
            if y.shape[0] != n_samples:
                raise ValueError(
                    f"View {m} has {y.shape[0]} samples instead of {n_samples}, "
                    "all views must have the same number of samples."
                )
            n_features.append(y.shape[1])
            feature_offsets.append(feature_offsets[-1] + n_features[-1])

            if isinstance(y, pd.DataFrame):
                logger.info("pd.DataFrame detected.")

                new_sample_names = y.index.tolist()
                if sample_names is None:
                    logger.info(
                        "Storing the index of the first view "
                        "as sample names and columns "
                        "of each dataframe feature names."
                    )
                    sample_names = new_sample_names
                if sample_names != new_sample_names:
                    logger.info(
                        "Sample names for view %s "
                        "do not match the sample names of view 0, "
                        "sorting names according to view 0.",
                        m,
                    )
                    y = y.loc[sample_names, :]
                feature_names.append(y.columns.tolist())
                numpy_observations.append(y.to_numpy())
            else:
                feature_names.append(None)
                numpy_observations.append(y)

        observations = numpy_observations

        n_views = len(n_features)
        if view_names is None:
            view_names = list(range(n_views))
        if sample_names is None:
            sample_names = list(range(n_samples))
        for m in range(n_views):
            if feature_names[m] is None:
                feature_names[m] = list(range(n_features[m]))
        self.n_samples = n_samples
        self.sample_names = sample_names
        self.n_features = n_features
        self.feature_offsets = feature_offsets
        self.feature_names = feature_names
        self.n_views = n_views
        self.view_names = view_names

        return observations

    def _setup_likelihoods(self, likelihoods):

        if likelihoods is None:
            likelihoods = ["normal" for _ in range(self.n_views)]
        if len(set(likelihoods)) > 1:
            logger.warning(
                "Different likelihoods for each view currently not supported, "
                "using %s for all views.",
                likelihoods[0],
            )
        return likelihoods

    def _setup_covariates(self, covariates):

        n_covariates = 0
        if covariates is not None:
            n_covariates = covariates.shape[1]

        covariate_names = None
        if isinstance(covariates, pd.DataFrame):
            logger.info("pd.DataFrame detected.")
            if self.n_samples != covariates.shape[0]:
                raise ValueError(
                    f"Number of observed samples for ({self.n_samples}) "
                    "does not match the number of samples "
                    f"for the covariates ({covariates.shape[0]})."
                )

            if self.sample_names != covariates.index.tolist():
                logger.info(
                    "Sample names for the covariates "
                    "do not match the sample names of the observations, "
                    "sorting names according to the observations."
                )
                covariates = covariates.loc[self.sample_names, :]
            covariate_names = covariates.columns.tolist()
            covariates = covariates.to_numpy()
        if covariate_names is None:
            covariate_names = list(range(n_covariates))

        self.n_covariates = n_covariates
        self.covariate_names = covariate_names
        return covariates

    def add_prior_masks(
        self, masks: List[Union[np.ndarray, pd.DataFrame]], confidence: float = 0.99
    ):
        """Generate local prior scales given a mask of feature sets.

        Parameters
        ----------
        masks : List[Union[np.ndarray, pd.DataFrame]]
            List of M K x D binary matrices of prior feature sets
        confidence : float, optional
            Confidence of the prior belief, larger more confident
            typical values 0.9, 0.95, 0.99, 0.999, by default 0.99
        """
        n_prior_factors = len(masks[0])
        if n_prior_factors > self.n_factors:
            logger.warning(
                "Prior mask informs more factors than the pre-defined `n_factors`. "
                "Updating `n_factors` to %s.",
                n_prior_factors,
            )
            self.n_factors = n_prior_factors

        if n_prior_factors < self.n_factors:
            logger.warning(
                "Prior mask informs fewer factors than the pre-defined `n_factors`. "
                "Informing only the first %s factors.",
                n_prior_factors,
            )
        factor_names = None
        prior_masks = []
        prior_scales = []
        for m, view_mask in enumerate(masks):
            if view_mask.shape[0] != self.n_factors:
                raise ValueError(
                    f"Mask {m} has {view_mask.shape[0]} factors "
                    f"instead of {self.n_factors}, "
                    "all masks must have the same number of factors."
                )

            if view_mask.shape[1] != self.n_features[m]:
                raise ValueError(
                    f"Mask {m} has {view_mask.shape[1]} features "
                    f"instead of {self.n_features[m]}, "
                    "each mask must match the number of features of its view."
                )
            if isinstance(view_mask, pd.DataFrame):
                logger.info("pd.DataFrame detected.")

                new_factor_names = view_mask.index.tolist()
                if factor_names is None:
                    logger.info("Storing the index of the first mask as factor names.")
                    factor_names = new_factor_names
                if factor_names != new_factor_names:
                    logger.info(
                        "Factor names for mask %s "
                        "do not match the factor names of mask 0, "
                        "sorting names according to mask 0.",
                        m,
                    )
                    view_mask = view_mask.loc[factor_names, :]

                if (
                    self.feature_names[m] is not None
                    and self.feature_names[m] != view_mask.columns.tolist()
                ):
                    logger.info(
                        "Feature names for mask %s "
                        "do not match the feature names of its corresponding view, "
                        "sorting names according to the view features.",
                        m,
                    )
                    view_mask = view_mask.loc[:, self.feature_names[m]]
            view_prior_scales = np.ones((self.n_factors, self.n_features[m]))
            view_prior_scales[: self.n_factors, :] = np.clip(
                view_mask.astype(np.float32) + (1.0 - confidence), 1e-4, 1.0
            )
            prior_masks.append(view_mask)
            prior_scales.append(view_prior_scales)

        if factor_names is None:
            factor_names = list(range(self.n_factors))
        self.factor_names = factor_names
        self.prior_masks = prior_masks
        self.prior_scales = prior_scales

    def _setup_model_guide(self, batch_size: int):
        """Setup model and guide.

        Parameters
        ----------
        batch_size : int
            Batch size when subsampling

        Returns
        -------
        bool
            Whether the build was successful
        """
        if not self._is_built:
            if self.prior_scales is None:
                logger.warning(
                    "No prior feature sets provided, running model uninformed."
                )

            self.model = MuVIModel(
                self.n_samples,
                n_subsamples=batch_size,
                n_features=self.n_features,
                n_factors=self.n_factors,
                n_covariates=self.n_covariates,
                likelihoods=self.likelihoods,
                device=self.device,
                **self.kwargs,
            )
            self.guide = MuVIGuide(self.model, device=self.device, **self.kwargs)
            self._is_built = True
        return self._is_built

    def _setup_optimizer(self, optimizer: str, learning_rate: float, n_iterations: int):
        """Setup SVI optimizer.

        Parameters
        ----------
        optimizer : str
            Optimizer as string, 'adam' or 'clipped'
        learning_rate : float
            Learning rate
        n_iterations : int
            Number of SVI iterations,
            needed to schedule learning rate decay

        Returns
        -------
        pyro.optim.PyroOptim
            pyro or torch optimizer object
        """

        optim = Adam({"lr": learning_rate, "betas": (0.95, 0.999)})
        if optimizer.lower() == "clipped":
            gamma = 0.1
            lrd = gamma ** (1 / n_iterations)
            optim = ClippedAdam({"lr": learning_rate, "lrd": lrd})

        self._optimizer = optim
        return self._optimizer

    def _setup_svi(
        self,
        batch_size: int,
        optimizer: pyro.optim.PyroOptim,
        n_particles: int,
        scale: bool = True,
    ):
        """Setup stochastic variational inference.

        Parameters
        ----------
        batch_size : int
            Batch size
        optimizer : pyro.optim.PyroOptim
            pyro or torch optimizer
        n_particles : int
            Number of particles/samples used to form the ELBO (gradient) estimators
        scale : bool, optional
            Whether to scale ELBO by 1/batch_size, by default True

        Returns
        -------
        pyro.infer.SVI
            pyro SVI object
        """
        scaler = 1.0
        if scale:
            scaler = 1.0 / batch_size

        svi = pyro.infer.SVI(
            model=pyro.poutine.scale(self.model, scale=scaler),
            guide=pyro.poutine.scale(self.guide, scale=scaler),
            optim=optimizer,
            loss=pyro.infer.TraceMeanField_ELBO(
                retain_graph=True,
                num_particles=n_particles,
                vectorize_particles=True,
            ),
        )

        self._svi = svi
        return self._svi

    def _setup_training_data(self):
        """Setup training components.
        Convert observations, covariates and prior scales to torch.Tensor and
        extract mask of missing values to mask SVI updates.

        Returns
        -------
        tuple
            Tuple of (obs, mask, covs, prior_scales)
        """
        train_obs = torch.cat(
            [torch.Tensor(y).to(self.device) for y in self.observations], 1
        )
        mask_obs = ~torch.isnan(train_obs)
        # replace all nans with zeros
        # self.presence mask takes care of gradient updates
        train_obs = torch.nan_to_num(train_obs)

        train_covs = None
        if self.covariates is not None:
            train_covs = torch.Tensor(self.covariates).to(self.device)

        train_prior_scales = None
        if self.prior_scales is not None:
            train_prior_scales = torch.cat(
                [torch.Tensor(ps).to(self.device) for ps in self.prior_scales], 1
            )

        return train_obs, mask_obs, train_covs, train_prior_scales

    def fit(
        self,
        batch_size: int = 0,
        n_iterations: int = 1000,
        n_particles: int = 20,
        learning_rate: float = 0.005,
        optimizer: str = "clipped",
        verbose: bool = True,
        callbacks: List[Callable] = None,
    ):
        """Perform inference.

        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default 0 (all samples)
        n_iterations : int, optional
            Number of iterations, by default 1000
        n_particles : int, optional
            Number of particles/samples used to form the ELBO (gradient) estimators,
            by default 20
        learning_rate : float, optional
            Learning rate, by default 0.005
        optimizer : str, optional
            Optimizer as string, 'adam' or 'clipped', by default "clipped"
        verbose : bool, optional
            Whether to log progress, by default 1
        callbacks : List[Callable], optional
            List of callbacks during training, by default None

        Returns
        -------
        tuple
            Tuple of (elbo history, whether training stopped early)
        """

        # if invalid or out of bounds set to n_samples
        if batch_size is None or not (0 < batch_size <= self.n_samples):
            batch_size = self.n_samples

        if batch_size < self.n_samples:
            logger.info("Using batches of size %s.", batch_size)
        else:
            logger.info("Using complete dataset.")

        logger.info("Preparing model and guide...")
        self._setup_model_guide(batch_size)
        logger.info("Preparing optimizer...")
        optimizer = self._setup_optimizer(optimizer, learning_rate, n_iterations)
        logger.info("Preparing SVI...")
        svi = self._setup_svi(batch_size, optimizer, n_particles, scale=True)
        logger.info("Preparing training data...")
        (
            train_obs,
            mask_obs,
            train_covs,
            train_prior_scales,
        ) = self._setup_training_data()
        logger.info("Starting training...")

        stop_early = False
        history = []
        pbar = range(n_iterations)
        if verbose > 0:
            pbar = tqdm(pbar)
            window_size = 10
        for iteration_idx in pbar:
            iteration_loss = svi.step(
                train_obs, mask_obs, train_covs, train_prior_scales
            )
            history.append(iteration_loss)
            if verbose > 0:
                if (
                    iteration_idx % window_size == 0
                    or iteration_idx == n_iterations - 1
                ):
                    pbar.set_postfix({"ELBO": iteration_loss})
            if callbacks is not None:
                # TODO: dont really like this, a bit sloppy
                stop_early = any([callback(history) for callback in callbacks])
                if stop_early:
                    break

        return history, stop_early

    @torch.no_grad()
    def _get_param(self, param_name: str):
        """Get parameters from the parameter store.

        Parameters
        ----------
        param_name : str
            Name of the parameter

        Returns
        -------
        np.ndarray
            Parameters
        """
        return self.guide.mode(param_name).cpu().detach().numpy()
        # return self.guide.median()[param_name].cpu().detach().numpy()

    @torch.no_grad()
    def _get_local_param(self, param_name: str, as_list: bool = False):
        """Get local parameters from the parameter store

        Parameters
        ----------
        param_name : str
            Name of the parameter
        as_list : bool, optional
            Whether to get a view-wise list of parameters, by default False

        Returns
        -------
        np.ndarray
            Parameters
        """
        local_param = self._get_param(param_name)
        if param_name == "sigma":
            local_param = local_param[None, :]

        if as_list:
            return [
                local_param[:, self.feature_offsets[m] : self.feature_offsets[m + 1]]
                for m in range(self.n_views)
            ]
        return local_param

    @torch.no_grad()
    def get_view_scale(self):
        """Get the view scales."""
        return self._get_param("view_scale")

    @torch.no_grad()
    def get_factor_scale(self):
        """Get the factor scales."""
        return self._get_param("factor_scale").T

    @torch.no_grad()
    def get_local_scale(self, as_list: bool = False):
        """Get the local scales."""
        return self._get_local_param("local_scale", as_list=as_list)

    @torch.no_grad()
    def get_caux(self, as_list: bool = False):
        """Get the c auxiliaries."""
        return self._get_local_param("caux", as_list=as_list)

    @torch.no_grad()
    def get_w(self, as_list: bool = False):
        """Get the factor loadings."""
        return self._get_local_param("w", as_list=as_list)

    @torch.no_grad()
    def get_beta(self, as_list: bool = False):
        """Get the beta coefficients."""
        return self._get_local_param("beta", as_list=as_list)

    @torch.no_grad()
    def get_z(self):
        """Get the factor scores."""
        return self._get_param("z")

    @torch.no_grad()
    def get_sigma(self, as_list: bool = False):
        """Get the marginal feature scales."""
        return self._get_local_param("sigma", as_list=as_list)


class MuVIModel(PyroModule):
    def __init__(
        self,
        n_samples: int,
        n_subsamples: int,
        n_features: List[int],
        n_factors: int,
        n_covariates: int,
        likelihoods: List[str],
        global_prior_scale: float = 1.0,
        device: bool = None,
    ):
        """MuVI generative model.

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_subsamples : int
            Number of subsamples (batch size)
        n_features : List[int]
            Number of features as list for each view
        n_factors : int
            Number of latent factors
        n_covariates : int
            Number of covariates
        likelihoods : List[str], optional
            List of likelihoods for each view,
            either 'normal' or 'bernoulli', by default None
        global_prior_scale : float, optional
            Determine the level of global sparsity, by default 1.0
        """
        super().__init__(name="MuVIModel")
        self.n_samples = n_samples
        self.n_subsamples = n_subsamples
        self.n_features = n_features
        self.feature_offsets = [0] + np.cumsum(self.n_features).tolist()
        self.n_views = len(self.n_features)
        self.n_factors = n_factors
        self.n_covariates = n_covariates
        self.likelihoods = likelihoods
        self.global_prior_scale = (
            1.0 if global_prior_scale is None else global_prior_scale
        )

        self.device = device
        self._plates = None

    @property
    def plates(self):
        """Get the sampling plates.

        Returns
        -------
        dict
        """
        if self._plates is None:
            plates = {
                "view": pyro.plate("view", self.n_views, dim=-1),
                "factor": pyro.plate("factor", self.n_factors, dim=-2),
                "feature": pyro.plate("feature", sum(self.n_features), dim=-1),
                "sample": pyro.plate(
                    "sample",
                    self.n_samples,
                    subsample_size=self.n_subsamples,
                    dim=-2,
                ),
            }
            if self.n_covariates > 0:
                plates["covariate"] = pyro.plate("covariate", self.n_covariates, dim=-2)
            self._plates = plates

        return self._plates

    def forward(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        covs: torch.Tensor = None,
        prior_scales: torch.Tensor = None,
    ):
        """Generate samples.

        Parameters
        ----------
        obs : torch.Tensor
            Observations to condition the model on
        mask : torch.Tensor
            Binary mask of missing data
        covs : torch.Tensor, optional
            Additional covariate matrix, by default None
        prior_scales : torch.Tensor, optional
            Local prior scales with prior information,
            by default None

        Returns
        -------
        dict
            Samples from each sampling site
        """

        output_dict = {}
        plates = self.plates

        with plates["view"]:
            output_dict["view_scale"] = pyro.sample(
                "view_scale", dist.HalfCauchy(torch.ones(1, device=self.device))
            )
            with plates["factor"]:
                output_dict["factor_scale"] = pyro.sample(
                    "factor_scale",
                    dist.HalfCauchy(torch.ones(1, device=self.device)),
                )

        with plates["feature"]:
            with plates["factor"]:
                output_dict["local_scale"] = pyro.sample(
                    "local_scale",
                    dist.HalfCauchy(torch.ones(1, device=self.device)),
                )
                output_dict["caux"] = pyro.sample(
                    "caux",
                    dist.InverseGamma(
                        0.5 * torch.ones(1, device=self.device),
                        0.5 * torch.ones(1, device=self.device),
                    ),
                )
                slab_scale = 1.0 if prior_scales is None else prior_scales
                c = slab_scale * torch.sqrt(output_dict["caux"])
                lmbda = torch.cat(
                    [
                        (
                            output_dict["local_scale"][
                                ...,
                                self.feature_offsets[m] : self.feature_offsets[m + 1],
                            ]
                            * output_dict["factor_scale"][..., m : m + 1]
                            * output_dict["view_scale"][..., m : m + 1]
                        )
                        for m in range(self.n_views)
                    ],
                    -1,
                )

                output_dict["w"] = pyro.sample(
                    "w",
                    dist.Normal(
                        torch.zeros(1, device=self.device),
                        (self.global_prior_scale * c * lmbda)
                        / torch.sqrt(c ** 2 + lmbda ** 2),
                    ),
                )

            if self.n_covariates > 0:
                with plates["covariate"]:
                    output_dict["beta"] = pyro.sample(
                        "beta",
                        dist.Normal(
                            torch.zeros(1, device=self.device),
                            torch.ones(1, device=self.device),
                        ),
                    )

            output_dict["sigma"] = pyro.sample(
                "sigma",
                dist.InverseGamma(
                    torch.ones(1, device=self.device),
                    torch.ones(1, device=self.device),
                ),
            )

        with plates["sample"] as indices:
            output_dict["z"] = pyro.sample(
                "z",
                dist.Normal(
                    torch.zeros(self.n_factors, device=self.device),
                    torch.ones(self.n_factors, device=self.device),
                ),
            )

            if self.n_subsamples < self.n_samples:
                indices = indices.to(self.device)
                obs = obs.index_select(0, indices)
                mask = mask.index_select(0, indices)
                covs = covs.index_select(0, indices)

            # TODO! extend to multiple different likelihoods
            y_loc = torch.matmul(output_dict["z"], output_dict["w"])
            if self.n_covariates > 0:
                y_loc = y_loc + torch.matmul(covs, output_dict["beta"])
            if self.likelihoods[0] == "normal":
                y_dist = dist.Normal(y_loc, torch.sqrt(output_dict["sigma"]))
            else:
                y_dist = dist.Bernoulli(logits=y_loc)
            with pyro.poutine.mask(mask=mask):
                output_dict["y"] = pyro.sample(
                    "y",
                    y_dist,
                    obs=obs,
                    infer={"is_auxiliary": True},
                )
        return output_dict


class MuVIGuide(PyroModule):
    def __init__(
        self,
        model,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
        device=None,
    ):
        """Approximate posterior.

        Parameters
        ----------
        model : object
            A MuVIModel object
        init_loc : float, optional
            Initial value for loc parameters, by default 0.0
        init_scale : float, optional
            Initial value for scale parameters, by default 0.1
        """
        super().__init__(name="MuVIGuide")
        self.model = model
        self.locs = PyroModule()
        self.scales = PyroModule()

        self.init_loc = init_loc
        self.init_scale = init_scale
        self.device = device

        self.setup()

    def _get_loc_and_scale(self, name: str):
        """Get loc and scale parameters.

        Parameters
        ----------
        name : str
            Name of the sampling site

        Returns
        -------
        tuple
            Tuple of (loc, scale)
        """
        site_loc = deep_getattr(self.locs, name)
        site_scale = deep_getattr(self.scales, name)
        return site_loc, site_scale

    def setup(self):
        """Setup parameters and sampling sites."""
        n_features = sum(self.model.n_features)
        n_views = self.model.n_views
        n_factors = self.model.n_factors

        site_to_shape = {
            "z": (self.model.n_samples, n_factors),
            "view_scale": n_views,
            "factor_scale": (n_factors, n_views),
            "local_scale": (n_factors, n_features),
            "caux": (n_factors, n_features),
            "w": (n_factors, n_features),
            "beta": (self.model.n_covariates, n_features),
            "sigma": n_features,
        }

        for name, shape in site_to_shape.items():
            deep_setattr(
                self.locs,
                name,
                PyroParam(
                    self.init_loc * torch.ones(shape, device=self.device),
                    constraints.real,
                ),
            )
            deep_setattr(
                self.scales,
                name,
                PyroParam(
                    self.init_scale * torch.ones(shape, device=self.device),
                    constraints.softplus_positive,
                ),
            )

    @torch.no_grad()
    def mode(self, name: str):
        """Get the MAP estimates.

        Parameters
        ----------
        name : str
            Name of the sampling site

        Returns
        -------
        torch.Tensor
            MAP estimate
        """
        loc, scale = self._get_loc_and_scale(name)
        mode = loc
        if name not in ["z", "w", "beta"]:
            mode = (loc - scale.pow(2)).exp()
        return mode.clone()

    def _sample_normal(self, name: str):
        return pyro.sample(name, dist.Normal(*self._get_loc_and_scale(name)))

    def _sample_log_normal(self, name: str):
        return pyro.sample(name, dist.LogNormal(*self._get_loc_and_scale(name)))

    def forward(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        covs: torch.Tensor = None,
        prior_scales: torch.Tensor = None,
    ):
        """Approximate posterior."""
        output_dict = {}

        plates = self.model.plates

        with plates["view"]:
            output_dict["view_scale"] = self._sample_log_normal("view_scale")
            with plates["factor"]:
                output_dict["factor_scale"] = self._sample_log_normal("factor_scale")

        with plates["feature"]:
            with plates["factor"]:
                output_dict["local_scale"] = self._sample_log_normal("local_scale")
                output_dict["caux"] = self._sample_log_normal("caux")
                output_dict["w"] = self._sample_normal("w")

            if self.model.n_covariates > 0:
                with plates["covariate"]:
                    output_dict["beta"] = self._sample_normal("beta")

            output_dict["sigma"] = self._sample_log_normal("sigma")

        with plates["sample"]:
            output_dict["z"] = self._sample_normal("z")
        return output_dict
