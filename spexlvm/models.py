"""Definition of the sparse factor analysis as a pyro module."""
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.infer import SVI
from pyro.infer.autoguide.guides import AutoNormal
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.optim import Adam
from tqdm import tqdm

from spexlvm import config

# logging stuff
logger = config.logger


class FactorAnalysis(PyroModule):
    """Standard factor analysis."""

    def __init__(self, n_features: int, n_factors: int, use_cuda=False, **kwargs):
        """Initialise vanilla factor analysis.

        Parameters
        ----------
        n_features : int
            Number of features (genes)
        n_factors : int
            Number of sparse factors to learn
        """
        super().__init__()

        self.n_features = n_features
        self.n_factors = n_factors

        self.sigma = PyroSample(dist.InverseGamma(0.5 * torch.ones(1), 0.5 * torch.ones(1)))
        # self.w = PyroSample(dist.Normal(torch.zeros(1), torch.ones(1)))
        self.w = PyroSample(lambda self: dist.Normal(torch.zeros(1), torch.ones(1)))

        self.x = PyroSample(
            dist.Normal(torch.zeros(self.n_factors), torch.ones(self.n_factors)).to_event(1)
        )

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

        self._guide = None
        self._svi = None
        self._is_built = False

    @property
    def guide(self):
        """Get the mean-field normal guide.

        Returns
        -------
        AutoNormal
            A normal auto guide from pyro.infer.autoguide.guides
        """
        if self._guide is None:
            self._guide = AutoNormal(self)
            # see https://github.com/pyro-ppl/numpyro/issues/855
            self.guide.scale_constraint = constraints.softplus_positive
        return self._guide

    @property
    def svi(self):
        """Initialise the Stochastic Variational Inference (SVI) algorithm.

        Returns
        -------
        SVI
            A pyro.infer.SVI object
        """
        if self._svi is None:
            self._svi = pyro.infer.SVI(
                model=self,
                guide=self.guide,
                optim=Adam({"lr": self.learning_rate, "betas": (0.95, 0.999)}),
                loss=pyro.infer.TraceMeanField_ELBO(),
                # loss=pyro.infer.Trace_ELBO(),
            )
        return self._svi

    @property
    def W(self):
        """Get the weight matrix of the model.

        Returns
        -------
        torch.Tensor
        """
        return pyro.param("_guide.locs.w")

    @property
    def X(self):
        """Get the factor scores for each datapoint.

        Returns
        -------
        torch.Tensor
        """
        return pyro.param("_guide.locs.x")

    def forward(self, n_samples: int, data=None):
        """Perform an iteration of the data generating process.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        data : torch.Tensor, optional
            Observations to condition the model on,
            needed for inference,
            by default None

        Returns
        -------
        The values of all the variables of the model
        """
        data_plate = pyro.plate("data", n_samples)
        feature_plate = pyro.plate("feature", self.n_features, dim=-1)
        factor_plate = pyro.plate("factor", self.n_factors, dim=-2)

        with feature_plate:
            sigma = self.sigma
        with factor_plate:
            with feature_plate:
                w = self.w

        with data_plate as indices:
            x = self.x
            if data is not None:
                samples = data.index_select(0, indices.to(data.device))
                if self.use_cuda:
                    samples = samples.cuda()

            y = pyro.sample(
                "y",
                dist.Normal(torch.mm(x, w), sigma).to_event(1),
                # obs=data,
                obs=samples,
                infer={"is_auxiliary": True},
            )

        # return y

    def _build(self, learning_rate: float, batch_size: int):

        if not self._is_built:
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self._is_built = True
        return self._is_built

    def fit(
        self,
        data,
        learning_rate: float = 0.0001,
        batch_size: int = 100,
        n_iterations: int = 10000,
        verbose: int = 1,
        callbacks=None,
    ):
        """Perform inference.

        Parameters
        ----------
        data : torch.Tensor
            Observations to condition the model on
        learning_rate : float, optional
            Learning rate during the stochastic VI, by default 0.0001
        batch_size : int, optional
            Size of batches, by default 100
        n_iterations : int, optional
            Number of iterations, by default 10000
        verbose : int, optional
            Verbosity level, by default 1
        callbacks : List[Callable], optional
            A callable that takes an iteration index and an iteration loss, by default None

        Returns
        -------
        list
            The loss history
        """
        # fix batch_size if invalid or out of bounds
        n_samples = data.shape[0]
        if batch_size is None or not (0 < batch_size <= n_samples):
            batch_size = n_samples

        self._build(learning_rate, batch_size)
        svi = self.svi

        if batch_size < n_samples:
            logger.info("Using batches of size %s", batch_size)
        else:
            logger.info("Using complete dataset")

        history = []
        pbar = range(n_iterations)
        if verbose > 0:
            pbar = tqdm(pbar)
            window_size = 10
        for iteration_idx in pbar:
            iteration_loss = svi.step(n_samples, data) / n_samples
            history.append(iteration_loss)
            if verbose > 0:
                if iteration_idx % window_size == 0 or iteration_idx == n_iterations - 1:
                    pbar.set_postfix({"ELBO": iteration_loss})
            if callbacks is not None:
                # TODO: dont really like this, a bit sloppy
                stop_early = any([callback(history) for callback in callbacks])
                if stop_early:
                    break

        return history


class SparseFactorAnalysis(FactorAnalysis):
    """Sparse FA with a regularised horseshoe prior."""

    def __init__(
        self,
        n_features: int,
        n_factors: int,
        global_prior_scale=0.1,
        local_prior_scales=None,
        **kwargs,
    ):
        """Initialise sparse factor analysis.

        Parameters
        ----------
        n_features : int
            Number of features (genes)
        n_factors : int
            Number of sparse factors to learn
        global_prior_scale : float, optional
            The prior on of the global scale,
            small values encourage the weights to be sparse,
            by default 0.1
        local_prior_scales : torch.Tensor, optional
            A matrix of positive values that serves
            as a prior information on the structure of annotated factors,
            by default None
        """
        super().__init__(n_features, n_factors, **kwargs)
        self.global_prior_scale = global_prior_scale
        self.local_prior_scales = local_prior_scales

        self.global_scale_variance = PyroSample(
            dist.InverseGamma(0.5 * torch.ones(1), 0.5 * torch.ones(1))
        )
        self.global_scale_noncentered = PyroSample(
            dist.FoldedDistribution(dist.Normal(torch.zeros(1), torch.ones(1)))
            # dist.HalfNormal(torch.ones(1))
        )
        self.local_scale_variance = PyroSample(
            dist.InverseGamma(0.5 * torch.ones(1), 0.5 * torch.ones(1))
        )
        self.local_scale_noncentered = PyroSample(
            dist.FoldedDistribution(dist.Normal(torch.zeros(1), torch.ones(1)))
            # dist.HalfNormal(torch.ones(1))
        )

        # caux from reg horseshoe paper
        self.slab_scale_variance = PyroSample(
            dist.InverseGamma(0.5 * torch.ones(1), 0.5 * torch.ones(1))
        )

        # needed for lazy initialisation from the local scales
        self.w_scale = torch.ones(1)
        self.w = PyroSample(lambda self: dist.Normal(torch.zeros(1), self.w_scale))

    def forward(self, n_samples: int, data=None):
        """Perform an iteration of the data generating process.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        data : torch.Tensor, optional
            Observations to condition the model on,
            needed for inference,
            by default None

        Returns
        -------
        The values of all the variables of the model
        """

        data_plate = pyro.plate("data", n_samples, subsample_size=self.batch_size)
        feature_plate = pyro.plate("feature", self.n_features, dim=-1)
        factor_plate = pyro.plate("factor", self.n_factors, dim=-2)

        gsv = self.global_scale_variance
        gsn = self.global_scale_noncentered
        tau = gsn * torch.sqrt(gsv) * self.global_prior_scale
        with feature_plate:
            sigma = self.sigma
        with factor_plate:
            with feature_plate:

                lsv = self.local_scale_variance
                lsn = self.local_scale_noncentered
                lmbda = lsn * torch.sqrt(lsv)
                # multiply with tau as suggested in the paper
                lmbda_tau = lmbda * tau

                scv = self.slab_scale_variance
                # inject prior on slabs, c from paper
                c = torch.sqrt(scv)
                if self.local_prior_scales is not None:
                    c = c * self.local_prior_scales
                # lambda tilde from paper, no need to square it since we need std not variance
                self.w_scale = (c * lmbda_tau) / (c + lmbda_tau)
                # if self.local_prior_scales is not None:
                #     self.w_scale *= self.local_prior_scales
                w = self.w

        with data_plate as indices:
            x = self.x
            if data is not None and len(indices) < data.shape[0]:
                data = data.index_select(0, indices.to(data.device))
                if self.use_cuda:
                    data = data.cuda()

            y = pyro.sample(
                "y",
                dist.Normal(torch.mm(x, w), sigma).to_event(1),
                # obs=data,
                obs=data,
                infer={"is_auxiliary": True},
            )

        # return y
