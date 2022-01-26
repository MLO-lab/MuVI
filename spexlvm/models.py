from typing import Callable, List

import anndata as ad
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer.autoguide.guides import AutoNormal
from pyro.nn import PyroModule
from pyro.optim import Adam, ClippedAdam
from tqdm import tqdm

from spexlvm import config

# logging stuff
logger = config.logger


class Spex(PyroModule):
    def __init__(
        self,
        n_features: int,
        n_annotated: int,
        n_sparse: int = 0,
        n_dense: int = 0,
        likelihood: str = "normal",
        global_prior_scale: float = 0.1,
        factor_scale_on: bool = False,
        use_cuda=False,
    ):
        """Initialise model.

        Parameters
        ----------
        n_features : int
            Number of features (genes)
        n_annotated : int
            Number of annotated factors based on a pathway mask
        n_sparse : int, optional
            Number of sparse unannotated factors, by default 0
        n_dense : int, optional
            Number of dense unannotated factors, by default 0
        likelihood : str, optional
            Likelihood, either "normal" or "bernoulli", by default "normal"
        global_prior_scale : float, optional
            The prior on of the global scale,
            small values encourage the weights to be sparse,
            by default 0.1
        factor_scale_on : bool, optional
            Whether to set a (regularised) horseshoe prior on the factors as well,
            by default False
        use_cuda : bool, optional
            Whether to use a GPU, by default False
        """
        super().__init__()

        self.n_features = n_features
        self.n_annotated = n_annotated
        self.n_sparse = n_sparse
        self.n_dense = n_dense
        self.n_factors = n_annotated + n_sparse + n_dense
        if likelihood is None:
            likelihood = "normal"
        self.likelihood = likelihood

        self.global_prior_scale = global_prior_scale
        self.factor_scale_on = factor_scale_on

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

        self.adata = None
        self.data = None
        self.presence_mask = None
        self.local_prior_scales = None
        self.n_samples = 0
        self.batch_size = 0

        self._guide = None
        self._svi = None
        self._is_built = False

    @property
    def guide(self):
        """Get the AutoNormal guide.

        Returns
        -------
        pyro.infer.autoguide.guides.AutoNormal
        """
        if self._guide is None:
            self._guide = AutoNormal(self, init_scale=0.01)
        return self._guide

    @property
    def svi(self):
        """Initialise the stochastic variational inference (SVI) trainer.

        Returns
        -------
        pyro.infer.SVI
        """
        if self._svi is None:

            scale = 1.0 / self.batch_size
            self._svi = pyro.infer.SVI(
                model=pyro.poutine.scale(self, scale=scale),
                guide=pyro.poutine.scale(self.guide, scale=scale),
                # model=self,
                # guide=self.guide,
                optim=self.optimizer,
                loss=pyro.infer.TraceMeanField_ELBO(
                    retain_graph=True,
                ),
            )
        return self._svi

    def get_plates(self, n_samples: int, subsample_size: int = None):
        """Get sampling plates.

        Parameters
        ----------
        n_samples : int
            Number of samples, usually len(data)
        subsample_size : int, optional
            Subsample size if relying on stochastic VI, by default None

        Returns
        -------
        dict
            Dictionary of plate names and pyro plates
        """
        return {
            "factor": pyro.plate("factor", self.n_factors, dim=-2),
            "feature": pyro.plate("feature", self.n_features, dim=-1),
            "sample": pyro.plate(
                "sample",
                n_samples,
                subsample_size=subsample_size,
                dim=-2,
            ),
        }

    def _get_param(self, param_name: str, as_map: bool = True):
        """Get parameters from the parameter store.

        Parameters
        ----------
        param_name : str
            Name of the parameter
        as_map : bool, optional
            Whether to get a MAP estimate, by default True

        Returns
        -------
        np.ndarray
            Parameters
        """
        map_output = self._guide(self.n_samples)
        if as_map:
            param_name = param_name.split(".")[-1]
            if param_name not in map_output.keys():
                return np.zeros((1, 1))
            param = map_output[param_name]
        else:
            if param_name not in pyro.get_param_store().keys():
                return np.zeros((1, 1))
            param = pyro.param(param_name)

        return param.cpu().detach().numpy()

    def get_global_scale(self):
        """Get the global scale of the model.

        Returns
        -------
        np.ndarray
        """
        return self._get_param("_guide.locs.global_scale", as_map=False)

    def get_factor_scale(self):
        """Get the factor scales of the model.

        Returns
        -------
        np.ndarray
        """
        return self._get_param("_guide.locs.factor_scale", as_map=False)

    def get_local_scale(self):
        """Get the local scales of the model.

        Returns
        -------
        np.ndarray
        """
        return self._get_param("_guide.locs.local_scale", as_map=False)

    def get_w(self):
        """Get the weights of the model.

        Returns
        -------
        np.ndarray
        """
        return self._get_param("_guide.locs.w", as_map=False)

    def get_x(self):
        """Get the factor scores for each datapoint.

        Returns
        -------
        np.ndarray
        """
        return self._get_param("_guide.locs.x", as_map=False)

    def get_precision(self):
        """Get the feature precisions of the model.

        Returns
        -------
        np.ndarray
        """
        return self._get_param("_guide.locs.precision", as_map=True)

    def mask_to_prior_scales(
        self,
        mask: np.ndarray,
        scale_confidence: float,
        sparse_prior_scale: float = 0.05,
        dense_prior_scale: float = 1.0,
    ):
        """Convert a boolean/binary mask into a prior for the local scales.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask
        scale_confidence : float
            User confidence about the prior information,
            common values are 0.9, 0.99, 0.999
        sparse_prior_scale : float, optional
            Prior local scales for the sparse factors, by default 0.05
        dense_prior_scale : float, optional
            Prior local scales for the dense factors, by default 1.0

        Returns
        -------
        torch.Tensor
            Local prior scales
        """

        prior_scales = np.ones((self.n_factors, self.n_features))
        prior_scales[: self.n_annotated, :] = np.clip(
            mask.astype(np.float32) + (1.0 - scale_confidence), 1e-3, dense_prior_scale
        )

        prior_scales[
            self.n_annotated : self.n_annotated + self.n_sparse, :
        ] = sparse_prior_scale
        prior_scales[
            self.n_annotated + self.n_sparse : self.n_factors, :
        ] = dense_prior_scale
        return torch.Tensor(prior_scales)

    def build(
        self,
        adata: ad.AnnData = None,
        data: np.ndarray = None,
        pathway_mask: np.ndarray = None,
        scale_confidence: float = 0.99,
    ):
        """Build model after initialisation.

        Parameters
        ----------
        adata : ad.AnnData, optional
            AnnData as training data, by default None
        data : np.ndarray, optional
            Numpy array as training data, by default None
        pathway_mask : np.ndarray, optional
            Binary mask of the pathways as prior information, by default None
        scale_confidence : float, optional
            User confidence about the prior information,
            common values are 0.9, 0.99, 0.999,
            by default 0.99

        Returns
        -------
        bool
            Whether the build was successful

        Raises
        ------
        ValueError
            Raised if neither adata nor data was passed during build
        """
        if adata is None and data is None:
            raise ValueError(
                "Invalid data! Both adata and data are None, "
                "please pass an AnnData object or a torch Tensor."
            )

        if not self._is_built:
            if adata is not None:
                self.adata = adata
                if data is not None:
                    logger.warning("Both adata and data passed, using adata only.")
                data = adata.X.copy()
                pathway_mask = None
                if "pathway_mask" in adata.varm.keys():
                    pathway_mask = adata.varm["pathway_mask"].T
            self.data = torch.Tensor(data)
            self.presence_mask = ~torch.isnan(self.data)
            if pathway_mask is not None:
                self.local_prior_scales = self.mask_to_prior_scales(
                    pathway_mask, scale_confidence=scale_confidence
                )
            self.n_samples = self.data.shape[0]
            # can be overwritten later during `fit`
            # only needed when generating data from the model
            self.batch_size = self.n_samples
            self._is_built = True

        return self._is_built

    def fit(
        self,
        batch_size: int = 0,
        n_iterations: int = 1000,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        verbose: int = 1,
        callbacks: List[Callable] = None,
    ):
        """Perform inference.

        Parameters
        ----------
        batch_size : int, optional
            Size of batches, by default 0
        n_iterations : int, optional
            Number of iterations, by default 1000
        learning_rate : float, optional
            Learning rate, by default 0.001
        optimizer : str, optional
            Optimiser as string, either "adam" or "clipped", by default "adam"
        verbose : int, optional
            Verbosity level during training, by default 1
        callbacks : List[Callable], optional
            A list of callables that takes the loss history as argument,
            by default None

        Returns
        -------
        tuple
            (The loss history, boolean flag whether training stopped early)
        """

        if self.data is None:
            logger.warning("No data available. Please build model before training.")

        # if invalid or out of bounds set to n_samples
        if batch_size is None or not (0 < batch_size <= self.n_samples):
            batch_size = self.n_samples
        self.batch_size = batch_size

        optim = Adam({"lr": learning_rate, "betas": (0.95, 0.999)})
        if optimizer.lower() == "clipped":
            gamma = 0.1
            lrd = gamma ** (1 / n_iterations)
            optim = ClippedAdam({"lr": learning_rate, "lrd": lrd})

        self.optimizer = optim
        svi = self.svi

        # replace all nans with zeros
        # self.presence mask takes care of gradient updates
        train_data = torch.nan_to_num(self.data)

        if self.batch_size < self.n_samples:
            logger.info("Using batches of size %s", batch_size)
        else:
            logger.info("Using complete dataset")

        stop_early = False
        history = []
        pbar = range(n_iterations)
        if verbose > 0:
            pbar = tqdm(pbar)
            window_size = 10
        for iteration_idx in pbar:
            iteration_loss = svi.step(self.n_samples, train_data)
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

    def forward(self, n_samples: int, data: torch.Tensor = None):
        """Perform an iteration of the data generating process.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        data : torch.Tensor, optional
            Observations to condition the model on,
            needed during inference,
            by default None

        Returns
        -------
        dict
            Dictonary of the random variables of the model
        """

        plates = self.get_plates(n_samples, min(n_samples, self.batch_size))

        global_scale = pyro.sample("global_scale", dist.HalfCauchy(torch.ones(1)))

        factor_scale = torch.ones(1)
        with plates["factor"]:
            if self.factor_scale_on:
                factor_scale = pyro.sample(
                    "factor_scale", dist.HalfCauchy(torch.ones(1))
                )

            with plates["feature"]:
                local_scale = pyro.sample("local_scale", dist.HalfCauchy(torch.ones(1)))

                slab_scale_variance = pyro.sample(
                    "slab_scale_variance",
                    dist.InverseGamma(0.5 * torch.ones(1), 0.5 * torch.ones(1)),
                )
                c = torch.sqrt(slab_scale_variance)

                if self.local_prior_scales is not None:
                    c = c * self.local_prior_scales
                local_scale = (c * local_scale) / (c + global_scale * local_scale)
                w = pyro.sample(
                    "w",
                    dist.Normal(
                        torch.zeros(1),
                        local_scale
                        * factor_scale
                        * global_scale
                        * self.global_prior_scale,
                    ),
                )

        with plates["feature"]:
            precision = pyro.sample(
                "precision",
                dist.Gamma(torch.ones(1), torch.ones(1)),
            )

        with plates["sample"] as indices:

            x = pyro.sample(
                "x",
                dist.Normal(torch.zeros(self.n_factors), torch.ones(self.n_factors)),
            )

            samples = data
            samples_mask = True
            if samples is not None:
                indices = indices.to(samples.device)
                samples = samples.index_select(0, indices)
                samples_mask = self.presence_mask.index_select(0, indices)

                if self.use_cuda:
                    samples = samples.cuda()
                    samples_mask = samples_mask.cuda()

            if self.likelihood == "normal":
                y_dist = dist.Normal(torch.mm(x, w), 1.0 / torch.sqrt(precision))
            else:
                y_dist = dist.Bernoulli(logits=torch.mm(x, w))
            with pyro.poutine.mask(mask=samples_mask):
                y = pyro.sample(
                    "y",
                    y_dist,
                    obs=samples,
                    infer={"is_auxiliary": True},
                )

        return {
            "global_scale": global_scale,
            "factor_scale": factor_scale,
            "local_scale": local_scale,
            "w": w,
            "x": x,
            "y": y,
            "precision": precision,
        }

    def fill_adata(self):
        """Fill AnnData object after training terminates.

        Returns
        -------
        ad.AnnData
            An updated version of self.adata

        Raises
        ------
        AttributeError
            Raised when an AnnData object was not passed during build
        """
        if self.adata is None:
            raise AttributeError(
                "Missing AnnData! No AnnData object passed during build."
            )

        columns = (
            self.adata.uns["pathway_names"]
            + [f"SPARSE_{i}" for i in range(self.n_sparse)]
            + [f"DENSE_{i}" for i in range(self.n_dense)]
        )
        self.adata.varm["W"] = pd.DataFrame(
            self.get_w().T,
            index=self.adata.var_names,
            columns=columns,
        )
        self.adata.obsm["X"] = pd.DataFrame(
            self.get_x(),
            index=self.adata.obs_names,
            columns=columns,
        )
        return self.adata
