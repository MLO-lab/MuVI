import logging
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pandas.api.types import is_string_dtype
from pyro.distributions import constraints
from pyro.infer.autoguide.guides import deep_getattr, deep_setattr
from pyro.nn import PyroModule, PyroParam
from pyro.optim import Adam, ClippedAdam
from tqdm import tqdm

from .index import _normalize_index

logger = logging.getLogger(__name__)

SINGLE_VIEW_TYPE = Union[np.ndarray, pd.DataFrame]
MULTI_VIEW_TYPE = Union[Dict[str, SINGLE_VIEW_TYPE], List[SINGLE_VIEW_TYPE]]


class MuVI(PyroModule):
    def __init__(
        self,
        observations: MULTI_VIEW_TYPE,
        prior_masks: MULTI_VIEW_TYPE = None,
        covariates: SINGLE_VIEW_TYPE = None,
        prior_confidence: float = None,
        n_factors: int = None,
        view_names: List[str] = None,
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

        self.observations = self._setup_observations(observations, view_names)
        self.prior_masks, self.prior_scales = self._setup_prior_masks(
            prior_masks, prior_confidence, n_factors
        )
        self.covariates = self._setup_covariates(covariates)
        self.likelihoods = self._setup_likelihoods(likelihoods)

        self.device = torch.device("cpu")
        if use_gpu and torch.cuda.is_available():
            logger.info("GPU available, running all computations on the GPU.")
            self.device = torch.device("cuda")
        self.to(self.device)

        self.kwargs = kwargs

        self._model = None
        self._guide = None
        self._informed = self.prior_masks is not None
        self._built = False
        self._trained = False
        self._cache = {}

    def _validate_index(self, idx):
        if not is_string_dtype(idx):
            logger.warning("Transforming to str index.")
            return idx.astype(str)
        return idx

    def _setup_observations(self, observations, view_names):
        if observations is None:
            raise ValueError(
                "Observations is None, please provide a valid list of observations."
            )

        n_views = len(observations)
        if n_views == 1:
            logger.warning("Running MuVI on a single view.")

        if view_names is None:
            logger.warning("No view names provided!")
            if isinstance(observations, list):
                logger.info(
                    "Setting the name of each view to `view_idx` for list observations."
                )
                view_names = [f"view_{m}" for m in range(n_views)]
            if isinstance(observations, dict):
                logger.info(
                    "Setting the view names to the sorted list "
                    "of dictonary keys in observations."
                )
                view_names = sorted(observations.keys())
        view_names = pd.Index(view_names)
        # if list convert to dict
        if isinstance(observations, list):
            observations = {vn: obs for vn, obs in zip(view_names, observations)}

        n_samples = observations[view_names[0]].shape[0]
        sample_names = None
        n_features = {vn: 0 for vn in view_names}
        feature_names = {vn: None for vn in view_names}
        for vn in view_names:
            y = observations[vn]
            if y.shape[0] != n_samples:
                raise ValueError(
                    f"View `{vn}` has {y.shape[0]} samples instead of {n_samples}, "
                    "all views must have the same number of samples."
                )
            n_features[vn] = y.shape[1]
            if isinstance(y, pd.DataFrame):
                logger.info("pd.DataFrame detected.")

                new_sample_names = y.index.copy()
                if sample_names is None:
                    logger.info(
                        "Storing the index of the view `%s` "
                        "as sample names and columns "
                        "of each dataframe feature names.",
                        vn,
                    )
                    sample_names = new_sample_names
                if any(sample_names != new_sample_names):
                    logger.info(
                        "Sample names for view `%s` "
                        "do not match the sample names of view `%s`, "
                        "sorting names according to view `%s`.",
                        vn,
                        view_names[0],
                        view_names[0],
                    )
                    observations[vn] = y.loc[sample_names, :]
                feature_names[vn] = y.columns.copy()

        if sample_names is None:
            logger.info("Setting the name of each sample to `sample_idx`.")
            sample_names = pd.Index([f"sample_{i}" for i in range(n_samples)])
        for vn in view_names:
            if feature_names[vn] is None:
                logger.info(
                    "Setting the name of each feature in `%s` to `%s_feature_idx`.",
                    vn,
                    vn,
                )
                feature_names[vn] = pd.Index(
                    [f"{vn}_feature_{j}" for j in range(n_features[vn])]
                )

        self.n_samples = n_samples
        self.n_views = n_views
        self.n_features = n_features
        self.sample_names = self._validate_index(sample_names)
        self.view_names = self._validate_index(view_names)
        self.feature_names = {
            vn: self._validate_index(fn) for vn, fn in feature_names.items()
        }

        # keep only numpy arrays
        return {
            vn: (obs.to_numpy() if isinstance(obs, pd.DataFrame) else obs)
            for vn, obs in observations.items()
        }

    def _setup_prior_masks(self, masks, confidence, n_factors):

        informed = masks is not None
        valid_n_factors = n_factors is not None and n_factors > 0
        if not informed and not valid_n_factors:
            raise ValueError(
                "Invalid latent configuration, "
                "please provide either a collection of prior masks, "
                "or set `n_factors` to a positive integer."
            )

        if not informed:
            self.n_factors = n_factors
            # TODO: duplicate line...see below
            self.factor_names = pd.Index([f"factor_{k}" for k in range(n_factors)])
            return None, None

        if confidence is None:
            logger.info(
                "No prior confidence provided, setting `prior_confidence` to 0.99."
            )
            confidence = 0.99

        if not (0 < confidence < 1.0):
            raise ValueError(
                "Invalid prior confidence, "
                "please provide a positive value less than 1."
            )

        # if list convert to dict
        if isinstance(masks, list):
            masks = {vn: view_mask for vn, view_mask in zip(self.view_names, masks)}
        informed_views = list(masks.keys())

        n_prior_factors = len(masks[informed_views[0]])
        if n_factors is None:
            n_factors = n_prior_factors

        if n_prior_factors > n_factors:
            logger.warning(
                "Prior mask informs more factors than the pre-defined `n_factors`. "
                "Updating `n_factors` to %s.",
                n_prior_factors,
            )
            n_factors = n_prior_factors

        if n_prior_factors < n_factors:
            logger.warning(
                "Prior mask informs fewer factors than the pre-defined `n_factors`. "
                "Informing only the first %s factors.",
                n_prior_factors,
            )

        factor_names = None
        for vn in self.view_names:
            # fill uninformed views
            # will only execute if a subset of views are being informed
            if vn not in masks:
                logger.info(
                    "Mask for view `%s` not found, assuming `%s` to be uninformed.",
                    vn,
                    vn,
                )
                masks[vn] = np.zeros((n_factors, self.n_features[vn]))
                continue
            view_mask = masks[vn]
            if view_mask.shape[0] != n_factors:
                raise ValueError(
                    f"Mask `{vn}` has {view_mask.shape[0]} factors "
                    f"instead of {n_factors}, "
                    "all masks must have the same number of factors."
                )
            if view_mask.shape[1] != self.n_features[vn]:
                raise ValueError(
                    f"Mask `{vn}` has {view_mask.shape[1]} features "
                    f"instead of {self.n_features[vn]}, "
                    "each mask must match the number of features of its view."
                )

            if isinstance(view_mask, pd.DataFrame):
                logger.info("pd.DataFrame detected.")

                new_factor_names = view_mask.index.copy()
                if factor_names is None:
                    logger.info(
                        "Storing the index of the mask `%s` as factor names.", vn
                    )
                    factor_names = new_factor_names
                if any(factor_names != new_factor_names):
                    logger.info(
                        "Factor names for mask `%s` "
                        "do not match the factor names of mask `%s`, "
                        "sorting names according to mask `%s`.",
                        vn,
                        informed_views[0],
                        informed_views[0],
                    )
                    masks[vn] = view_mask.loc[factor_names, :]

                if any(self.feature_names[vn] != view_mask.columns):
                    logger.info(
                        "Feature names for mask `%s` "
                        "do not match the feature names of its corresponding view, "
                        "sorting names according to the view features.",
                        vn,
                    )
                    masks[vn] = view_mask.loc[:, self.feature_names[vn]]

        if factor_names is None:
            factor_names = pd.Index([f"factor_{k}" for k in range(n_factors)])
        self.factor_names = self._validate_index(factor_names)

        # keep only numpy arrays
        prior_masks = {
            vn: (
                vm.to_numpy().astype(bool)
                if isinstance(vm, pd.DataFrame)
                else vm.astype(bool)
            )
            for vn, vm in masks.items()
        }
        prior_scales = {
            vn: np.clip(vm.astype(np.float32) + (1.0 - confidence), 1e-4, 1.0)
            for vn, vm in prior_masks.items()
        }
        self.n_factors = n_factors
        return prior_masks, prior_scales

    def _setup_likelihoods(self, likelihoods):

        if likelihoods is None:
            likelihoods = ["normal" for _ in range(self.n_views)]
        if isinstance(likelihoods, list):
            likelihoods = {self.view_names[i]: ll for i, ll in enumerate(likelihoods)}
        likelihoods = {vn: likelihoods.get(vn, "normal") for vn in self.view_names}
        if len(set(likelihoods.values())) > 1:
            # TODO! fix
            logger.warning(
                "Different likelihoods for each view currently not supported, "
                "using `%s` for all views.",
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

            if any(self.sample_names != covariates.index):
                logger.info(
                    "Sample names for the covariates "
                    "do not match the sample names of the observations, "
                    "sorting names according to the observations."
                )
                covariates = covariates.loc[self.sample_names, :]
            covariate_names = covariates.columns.copy()
            covariates = covariates.to_numpy()
        if covariate_names is None:
            covariate_names = pd.Index([f"covariate_{k}" for k in range(n_covariates)])

        self.n_covariates = n_covariates
        self.covariate_names = covariate_names
        return covariates

    def _get_view_attr(
        self,
        attr,
        view_idx,
        feature_idx,
        other_idx,
        other_names,
        as_df,
    ):
        if view_idx is None and isinstance(feature_idx, dict):
            view_idx = []
            for key in feature_idx.keys():
                vi = key
                if isinstance(vi, int):
                    vi = self.view_names[vi]
                view_idx.append(vi)
        if view_idx is None:
            raise IndexError(
                "Invalid indices, `view_idx` is None "
                "and `feature_idx` is not a dictionary."
            )
        # get relevant view names
        view_names = _normalize_index(view_idx, self.view_names, as_idx=False)
        if isinstance(feature_idx, (str, int)):
            if feature_idx == "all":
                feature_idx = ["all" for _ in range(len(view_names))]
            else:
                feature_idx = [feature_idx]

        # convert to list if any list-like indexing
        if isinstance(feature_idx, (list, np.ndarray, pd.Index)):
            feature_idx = list(feature_idx)

        # check if valid combination of indices
        if len(view_names) == 1 and len(feature_idx) != 1:
            if isinstance(feature_idx[0], list):
                logger.warning(
                    "`feature_idx` suggests indices for more than one view, "
                    "keeping only the first list of indices."
                )
                feature_idx = feature_idx[0]
            feature_idx = [feature_idx]

        if len(view_names) != len(feature_idx):
            logger.warning(
                "`view_idx` does not match the keys of `feature_idx`, "
                "`view_idx` has precedence over `feature_idx`."
            )

        if isinstance(feature_idx, list):
            feature_idx = {vn: feature_idx[m] for m, vn in enumerate(view_names)}

        if isinstance(feature_idx, dict):
            feature_idx = {
                (self.view_names[k] if isinstance(k, int) else k): v
                for k, v in feature_idx.items()
            }

        # normalise dictionary
        feature_idx = {
            vn: _normalize_index(feature_idx[vn], self.feature_names[vn])
            for vn in view_names
        }
        other_idx = _normalize_index(other_idx, other_names)

        attr = {vn: attr[vn][other_idx, :][:, feature_idx[vn]] for vn in view_names}
        if as_df:
            attr = {
                vn: pd.DataFrame(
                    attr[vn],
                    index=other_names[other_idx],
                    columns=self.feature_names[vn][feature_idx[vn]],
                )
                for vn in view_names
            }
        return attr

    def _get_shared_attr(
        self,
        attr,
        sample_idx,
        other_idx,
        other_names,
        as_df,
    ):
        sample_idx = _normalize_index(sample_idx, self.sample_names)
        other_idx = _normalize_index(other_idx, other_names)
        attr = attr[sample_idx, :][:, other_idx]
        if as_df:
            attr = pd.DataFrame(
                attr,
                index=self.sample_names[sample_idx],
                columns=other_names[other_idx],
            )
        return attr

    def get_observations(
        self,
        view_idx="all",
        sample_idx="all",
        feature_idx="all",
        as_df: bool = False,
    ):
        return self._get_view_attr(
            self.observations,
            view_idx,
            feature_idx,
            other_idx=sample_idx,
            other_names=self.sample_names,
            as_df=as_df,
        )

    def get_prior_masks(
        self,
        view_idx="all",
        factor_idx="all",
        feature_idx="all",
        as_df: bool = False,
    ):
        return self._get_view_attr(
            self.prior_masks,
            view_idx,
            feature_idx,
            other_idx=factor_idx,
            other_names=self.factor_names,
            as_df=as_df,
        )

    def get_factor_loadings(
        self,
        view_idx="all",
        factor_idx="all",
        feature_idx="all",
        as_df: bool = False,
    ):
        ws = self._guide.get_w(as_list=True)
        return self._get_view_attr(
            {vn: ws[m] for m, vn in enumerate(self.view_names)},
            view_idx,
            feature_idx,
            other_idx=factor_idx,
            other_names=self.factor_names,
            as_df=as_df,
        )

    def get_covariate_coefficients(
        self,
        view_idx="all",
        cov_idx="all",
        feature_idx="all",
        as_df: bool = False,
    ):
        betas = self._guide.get_beta(as_list=True)
        return self._get_view_attr(
            {vn: betas[m] for m, vn in enumerate(self.view_names)},
            view_idx,
            feature_idx,
            other_idx=cov_idx,
            other_names=self.covariate_names,
            as_df=as_df,
        )

    def get_factor_scores(
        self, sample_idx="all", factor_idx="all", as_df: bool = False
    ):
        return self._get_shared_attr(
            self._guide.get_z(),
            sample_idx,
            other_idx=factor_idx,
            other_names=self.factor_names,
            as_df=as_df,
        )

    def get_covariates(self, sample_idx="all", cov_idx="all", as_df: bool = False):
        return self._get_shared_attr(
            self.covariates,
            sample_idx,
            other_idx=cov_idx,
            other_names=self.covariate_names,
            as_df=as_df,
        )

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
        if not self._built:
            if not self._informed:
                logger.warning(
                    "No prior feature sets provided, running model uninformed."
                )

            self._model = MuVIModel(
                self.n_samples,
                n_subsamples=batch_size,
                n_features=[self.n_features[vn] for vn in self.view_names],
                n_factors=self.n_factors,
                n_covariates=self.n_covariates,
                likelihoods=[self.likelihoods[vn] for vn in self.view_names],
                device=self.device,
                **self.kwargs,
            )
            self._guide = MuVIGuide(self._model, device=self.device, **self.kwargs)
            self._built = True
        return self._built

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
            model=pyro.poutine.scale(self._model, scale=scaler),
            guide=pyro.poutine.scale(self._guide, scale=scaler),
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
            [
                torch.Tensor(self.observations[vn]).to(self.device)
                for vn in self.view_names
            ],
            1,
        )
        mask_obs = ~torch.isnan(train_obs)
        # replace all nans with zeros
        # self.presence mask takes care of gradient updates
        train_obs = torch.nan_to_num(train_obs)

        train_covs = None
        if self.covariates is not None:
            train_covs = torch.Tensor(self.covariates).to(self.device)

        train_prior_scales = None
        if self._informed:
            train_prior_scales = torch.cat(
                [
                    torch.Tensor(self.prior_scales[vn]).to(self.device)
                    for vn in self.view_names
                ],
                1,
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
                        / torch.sqrt(c**2 + lmbda**2),
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

    @torch.no_grad()
    def _get_map_estimate(self, param_name: str, as_list: bool):
        param = self.mode(param_name).cpu().detach().numpy()
        if param_name == "sigma":
            param = param[None, :]

        if as_list:
            return [
                param[
                    :, self.model.feature_offsets[m] : self.model.feature_offsets[m + 1]
                ]
                for m in range(self.model.n_views)
            ]
        return param

    def get_sigma(self, as_list: bool = False):
        """Get the marginal feature scales."""
        return self._get_map_estimate("sigma", as_list=as_list)

    def get_view_scale(self):
        """Get the view scales."""
        return self._get_map_estimate("view_scale", False)

    def get_factor_scale(self):
        """Get the factor scales."""
        return self._get_map_estimate("factor_scale", False).T

    def get_local_scale(self, as_list: bool = False):
        """Get the local scales."""
        return self._get_map_estimate("local_scale", as_list=as_list)

    def get_caux(self, as_list: bool = False):
        """Get the c auxiliaries."""
        return self._get_map_estimate("caux", as_list=as_list)

    def get_z(self):
        """Get the factor scores."""
        return self._get_map_estimate("z", False)

    def get_w(self, as_list: bool = False):
        """Get the factor loadings."""
        return self._get_map_estimate("w", as_list=as_list)

    def get_beta(self, as_list: bool = False):
        """Get the beta coefficients."""
        return self._get_map_estimate("beta", as_list=as_list)

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
