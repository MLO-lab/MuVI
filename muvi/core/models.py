import logging
import time

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch

from pandas.api.types import is_string_dtype
from pyro.distributions import constraints
from pyro.infer.autoguide.guides import deep_getattr
from pyro.infer.autoguide.guides import deep_setattr
from pyro.nn import PyroModule
from pyro.nn import PyroParam
from pyro.optim import Adam
from pyro.optim import ClippedAdam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from muvi.core.index import _normalize_index


logger = logging.getLogger(__name__)

Index = Union[int, str, List[int], List[str], np.ndarray, pd.Index]
SingleView = Union[np.ndarray, pd.DataFrame]
MultiView = Union[Dict[str, SingleView], List[SingleView]]


class DictDataset(Dataset):
    """Dictionary based PyTorch dataset."""

    def __init__(self, tensor_dict, idx_key="sample_idx", **kwargs):
        self.tensor_dict = tensor_dict
        self.idx_key = idx_key
        # just stores them
        self.kwargs = kwargs
        self._len = tensor_dict[next(iter(tensor_dict))].shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        item = {self.idx_key: index}
        for k, v in self.tensor_dict.items():
            item[k] = v[index]
        return item


class MuVI(PyroModule):
    def __init__(
        self,
        observations: MultiView,
        prior_masks: Optional[MultiView] = None,
        covariates: Optional[SingleView] = None,
        prior_confidence: Union[float, str] = 0.99,
        n_factors: Optional[int] = None,
        view_names: Optional[List[str]] = None,
        likelihoods: Optional[Union[Dict[str, str], List[str]]] = None,
        reg_hs: bool = True,
        nmf: Optional[Union[Dict[str, bool], List[bool]]] = None,
        device: str = "cuda",
    ):
        """MuVI module.

        Parameters
        ----------
        observations : MultiView
            Collection of observations as list or dict.
        prior_masks : MultiView, optional
            Prior feature sets as a collection of binary masks,
            by default None
        covariates : SingleView, optional
            Additional observed covariates, by default None
        prior_confidence : float or string
            Confidence of prior belief from 0 to 1 (exclusive),
            typical values are 'low' (0.95), 'med' (0.99) and 'high' (0.999),
            by default 0.99
        n_factors : int, optional
            Number of latent factors,
            can be omitted when providing prior masks,
            by default None
        view_names : List[str], optional
            List of names for each view,
            determines view order as well,
            by default None
        likelihoods : Union[Dict[str, str], List[str]], optional
            Likelihoods for each view,
            either "normal" or "bernoulli",
            by default None (all "normal")
        reg_hs : bool, optional
            Whether to use the regularized version of HS,
            by default True,
            only relevant when NOT using informative priors
        nmf : Union[Dict[str, bool], List[bool]], optional
            Whether to use non-negative matrix factorization,
            by default False
        device : str, optional
            Device to run computations on, by default "cuda" (GPU)
        """
        super().__init__(name="MuVI")
        self.observations = self._setup_observations(observations, view_names)
        self.prior_masks, self.prior_scales = self._setup_prior_masks(
            prior_masks, prior_confidence, n_factors
        )
        self.covariates = self._setup_covariates(covariates)
        self.likelihoods = self._setup_likelihoods(likelihoods)
        self.nmf = self._setup_nmf(nmf)

        self.reg_hs = reg_hs
        self._informed = self.prior_masks is not None
        if not self.reg_hs and self._informed:
            logger.warning(
                "Informative priors require regularized horseshoe, "
                "setting `reg_hs` to True."
            )
            self.reg_hs = True

        self.device = self._setup_device(device)
        self.to(self.device)

        self._model = None
        self._guide = None
        self._built = False
        self._trained = False
        self._training_log: Dict[str, Any] = {}
        self._cache = None

    @property
    def _factor_signs(self):
        signs = 1.0
        # only compute signs if model is trained
        # and there is no nmf constraint
        if self._trained and not any(self.nmf.values()):
            w = self._guide.get_w()
            # TODO: looking at top loadings works better than including all
            # 100 feels a bit arbitrary though
            w = np.array(
                list(map(lambda x, y: y[x], np.argsort(-np.abs(w), axis=1)[:, :100], w))
            )
            signs = (w.sum(axis=1) > 0) * 2 - 1

        return pd.Series(signs, index=self.factor_names, dtype=np.float32)

    def _setup_device(self, device):
        cuda_available = torch.cuda.is_available()

        try:
            mps_available = torch.backends.mps.is_available()
        except AttributeError:
            mps_available = False

        device = str(device).lower()
        if ("cuda" in device and not cuda_available) or (
            device == "mps" and not mps_available
        ):
            logger.warning(f"`{device}` not available...")
            device = "cpu"

        logger.info(f"Running all computations on `{device}`.")
        return torch.device(device)

    def _validate_index(self, idx):
        if not is_string_dtype(idx):
            logger.warning("Transforming to str index.")
            return idx.astype(str)
        return idx

    def _setup_views(self, observations, view_names):
        if view_names is None:
            logger.warning("No view names provided!")
            if isinstance(observations, list):
                logger.info(
                    "Setting the name of each view to `view_idx` for list observations."
                )
                view_names = [f"view_{m}" for m in range(len(observations))]
            if isinstance(observations, dict):
                logger.info(
                    "Setting the view names to the sorted list "
                    "of dictonary keys in observations."
                )
                view_names = sorted(observations.keys())

        # if list convert to dict
        if isinstance(observations, list):
            observations = dict(zip(view_names, observations))

        _view_names = []
        for vn in view_names:
            if vn not in observations:
                logger.warning(f"View `{vn}` not found in observations, skipping.")
            else:
                _view_names.append(vn)
        view_names = _view_names
        if len(view_names) == 0:
            raise ValueError(
                "No views found, check if `view_names` matches the dictionary keys!"
            )

        if len(view_names) < len(observations):
            logger.warning(
                "Number of views is larger than the length of `view_names`, "
                "using only the subset of views defined in `view_names`."
            )

        observations = {vn: observations[vn] for vn in view_names}

        n_views = len(observations)
        if n_views == 1:
            logger.warning("Running MuVI on a single view.")

        view_names = pd.Index(view_names)

        self.n_views = n_views
        self.view_names = self._validate_index(view_names)
        return observations

    def _setup_samples(self, observations):
        n_samples = 0
        sample_names = None

        for vn in self.view_names:
            view_obs = observations[vn]
            view_n_samples = view_obs.shape[0]

            if n_samples == 0:
                n_samples = view_n_samples

            if n_samples != view_n_samples:
                raise ValueError(
                    f"View `{vn}` has {view_n_samples} samples "
                    f"instead of {n_samples}, "
                    "all views must have the same number of samples."
                )

            if isinstance(view_obs, pd.DataFrame):
                logger.info("pd.DataFrame detected.")

                view_sample_names = view_obs.index.copy()

                if sample_names is None:
                    logger.info(
                        f"Storing the index of the view `{vn}` as sample names."
                    )
                    sample_names = view_sample_names

                if any(sample_names != view_sample_names):
                    logger.info(
                        f"Sample names for view `{vn}` do not match "
                        f"the sample names of view `{self.view_names[0]}`, "
                        f"sorting names according to view `{self.view_names[0]}`."
                    )
                    view_obs = view_obs.loc[sample_names, :]

        if sample_names is None:
            logger.info("Setting the name of each sample to `sample_idx`.")
            sample_names = pd.Index([f"sample_{i}" for i in range(n_samples)])

        self.sample_names = self._validate_index(sample_names)
        self.n_samples = n_samples

        return observations

    def _setup_features(self, observations):
        n_features = {vn: 0 for vn in self.view_names}
        feature_names = {vn: None for vn in self.view_names}

        for vn in self.view_names:
            view_obs = observations[vn]
            n_features[vn] = view_obs.shape[1]
            if isinstance(view_obs, pd.DataFrame):
                logger.info("pd.DataFrame detected.")
                feature_names[vn] = view_obs.columns.copy()
            if feature_names[vn] is None:
                logger.info(
                    f"Setting the name of each feature in `{vn}` to `{vn}_feature_idx`."
                )
                feature_names[vn] = pd.Index(
                    [f"{vn}_feature_{j}" for j in range(n_features[vn])]
                )

            feature_names[vn] = self._validate_index(feature_names[vn])

        self.feature_names = feature_names
        self.n_features = n_features

        return observations

    def _setup_observations(self, observations, view_names):
        if observations is None:
            raise ValueError(
                "Observations is None, please provide a valid list of observations."
            )

        observations = self._setup_views(observations, view_names)
        # from now on only working with nested dictonaries
        observations = self._setup_samples(observations)
        observations = self._setup_features(observations)

        # keep only numpy arrays, convert to np.float32 dtypes
        return {
            vn: (
                obs.to_numpy(dtype=np.float32)
                if isinstance(obs, pd.DataFrame)
                else np.array(obs, dtype=np.float32)
            )
            for vn, obs in observations.items()
        }

    def _setup_prior_masks(self, masks, confidence, n_factors):
        informed = masks is not None and len(masks) > 0
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

        if isinstance(confidence, str):
            try:
                confidence = {"low": 0.95, "med": 0.99, "high": 0.999}[
                    confidence.lower()
                ]
            except KeyError as e:
                logger.error(e)
                raise

        if not (0 < confidence < 1.0):
            raise ValueError(
                "Invalid prior confidence, please provide a positive value less than 1."
            )

        # if list convert to dict
        if isinstance(masks, list):
            masks = {self.view_names[m]: mask for m, mask in enumerate(masks)}

        informed_views = [vn for vn in self.view_names if vn in masks]

        n_prior_factors = masks[informed_views[0]].shape[0]
        if n_factors is None:
            n_factors = n_prior_factors

        if n_prior_factors > n_factors:
            logger.warning(
                f"Prior mask informs more factors ({n_prior_factors}) "
                f"than the pre-defined `n_factors` ({n_factors}). "
                f"Updating `n_factors` to {n_prior_factors}."
            )
            n_factors = n_prior_factors

        n_dense_factors = 0
        if n_prior_factors < n_factors:
            logger.warning(
                f"Prior mask informs fewer factors ({n_prior_factors}) "
                f"than the pre-defined `n_factors` ({n_factors}). "
                f"Informing only the first {n_prior_factors} factors, "
                "the rest remains uninformed."
            )
            # extend all prior masks with additional uninformed factors
            n_dense_factors = n_factors - n_prior_factors

        factor_names = None
        for vn in self.view_names:
            # fill uninformed views
            # will only execute if a subset of views are being informed
            if vn not in masks:
                logger.info(
                    f"Mask for view `{vn}` not found, assuming `{vn}` to be uninformed."
                )
                masks[vn] = np.zeros((n_prior_factors, self.n_features[vn])).astype(
                    bool
                )
                continue
            view_mask = masks[vn]
            if view_mask.shape[0] != n_prior_factors:
                raise ValueError(
                    f"Mask `{vn}` has {view_mask.shape[0]} factors "
                    f"instead of {n_prior_factors}, "
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

                mask_factor_names = view_mask.index.copy()
                if factor_names is None:
                    logger.info(
                        f"Storing the index of the mask `{vn}` as factor names."
                    )
                    factor_names = mask_factor_names
                if any(factor_names != mask_factor_names):
                    logger.info(
                        f"Factor names for mask `{vn}` "
                        f"do not match the factor names of mask `{informed_views[0]}`, "
                        f"sorting names according to mask `{informed_views[0]}`."
                    )
                    masks[vn] = view_mask.loc[factor_names, :]

                if any(self.feature_names[vn] != view_mask.columns):
                    logger.info(
                        f"Feature names for mask `{vn}` "
                        "do not match the feature names of its corresponding view, "
                        "sorting names according to the view features."
                    )
                    masks[vn] = view_mask.loc[:, self.feature_names[vn]]

        if factor_names is None:
            factor_names = [f"factor_{k}" for k in range(n_prior_factors)]
        if n_dense_factors > 0:
            factor_names = list(factor_names) + [
                f"dense_{k}" for k in range(n_dense_factors)
            ]
        self.factor_names = self._validate_index(pd.Index(factor_names))

        # keep only numpy arrays
        prior_masks = {
            vn: (
                vm.to_numpy().astype(bool)
                if isinstance(vm, pd.DataFrame)
                else vm.astype(bool)
            )
            for vn, vm in masks.items()
        }
        # add dense factors if necessary
        if n_dense_factors > 0:
            prior_masks = {
                vn: np.concatenate(
                    [vm, np.ones((n_dense_factors, self.n_features[vn])).astype(bool)]
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
            likelihoods = {self.view_names[m]: ll for m, ll in enumerate(likelihoods)}
        likelihoods = {vn: likelihoods.get(vn, "normal") for vn in self.view_names}
        logger.info(f"Likelihoods set to `{likelihoods}`.")
        return likelihoods

    def _setup_nmf(self, nmf):
        if nmf is None:
            nmf = [False for _ in range(self.n_views)]
        if isinstance(nmf, bool):
            nmf = [nmf for _ in range(self.n_views)]
        if isinstance(nmf, list):
            nmf = {self.view_names[m]: _nmf for m, _nmf in enumerate(nmf)}
        nmf = {vn: nmf.get(vn, False) for vn in self.view_names}
        logger.info(f"NMF set to `{nmf}`.")
        return nmf

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

    def _raise_untrained_error(self):
        if not self._trained:
            raise AttributeError(
                "Requested attribute cannot be found on an untrained model! "
                "Run the `fit` method to train a MuVI model."
            )
        return True

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
            for key in feature_idx:
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
        shared_idx,
        shared_names,
        other_idx,
        other_names,
        as_df,
    ):
        shared_idx = _normalize_index(shared_idx, shared_names)
        other_idx = _normalize_index(other_idx, other_names)
        attr = attr[shared_idx, :][:, other_idx]
        if as_df:
            attr = pd.DataFrame(
                attr,
                index=shared_names[shared_idx],
                columns=other_names[other_idx],
            )
        return attr

    def _get_sample_attr(
        self,
        attr,
        sample_idx,
        other_idx,
        other_names,
        as_df,
    ):
        return self._get_shared_attr(
            attr, sample_idx, self.sample_names, other_idx, other_names, as_df
        )

    def get_observations(
        self,
        view_idx: Index = "all",
        sample_idx: Index = "all",
        feature_idx: Union[Index, List[Index], Dict[str, Index]] = "all",
        as_df: bool = False,
    ):
        """Get observations.

        Parameters
        ----------
        view_idx : Index, optional
            View index, by default "all"
        sample_idx : Index, optional
            Sample index, by default "all"
        feature_idx : Union[Index, List[Index], Dict[str, Index]], optional
            Feature index, by default "all"
        as_df : bool, optional
            Whether to return a pandas dataframe,
            by default False

        Returns
        -------
        dict
            Dictionary with view names as keys,
            and np.ndarray or pd.DataFrame as values.
        """
        return self._get_view_attr(
            self.observations,
            view_idx,
            feature_idx,
            other_idx=sample_idx,
            other_names=self.sample_names,
            as_df=as_df,
        )

    def _mode(self, param, likelihood):
        if likelihood == "bernoulli":
            return (param > 0.5).astype(np.int32)
        return param

    def get_reconstructed(
        self,
        view_idx: Index = "all",
        sample_idx: Index = "all",
        feature_idx: Union[Index, List[Index], Dict[str, Index]] = "all",
        factor_idx: Index = "all",
        cov_idx: Index = "all",
        as_df: bool = False,
    ):
        """Get reconstructed observations.

        Parameters
        ----------
        view_idx : Index, optional
            View index, by default "all"
        sample_idx : Index, optional
            Sample index, by default "all"
        feature_idx : Union[Index, List[Index], Dict[str, Index]], optional
            Feature index, by default "all"
        factor_idx : Index, optional
            Factor index, by default "all"
        cov_idx : Index, optional
            Covariate index, by default "all"
        as_df : bool, optional
            Whether to return a pandas dataframe,
            by default False

        Returns
        -------
        dict
            Dictionary with view names as keys,
            and np.ndarray or pd.DataFrame as values.
        """
        obs_hat = {
            vn: np.zeros((self.n_samples, self.n_features[vn]))
            for vn in _normalize_index(view_idx, self.view_names, as_idx=False)
        }

        if self.n_factors > 0:
            factor_scores = self.get_factor_scores(sample_idx, factor_idx)
            factor_loadings = self.get_factor_loadings(
                view_idx, factor_idx, feature_idx
            )
            obs_hat = {
                vn: obs + (factor_scores @ factor_loadings[vn])
                for vn, obs in obs_hat.items()
            }

        if self.n_covariates > 0:
            covariates = self.get_covariates(sample_idx, cov_idx)
            cov_coefficients = self.get_covariate_coefficients(
                view_idx, cov_idx, feature_idx
            )
            obs_hat = {
                vn: obs + (covariates @ cov_coefficients[vn])
                for vn, obs in obs_hat.items()
            }

        for vn in obs_hat:
            obs_hat[vn] = self._mode(obs_hat[vn], self.likelihoods[vn])

        if as_df:
            obs_hat = {
                vn: pd.DataFrame(
                    obs, index=self.sample_names, columns=self.feature_names[vn]
                )
                for vn, obs in obs_hat.items()
            }

        return obs_hat

    def get_imputed(
        self,
        view_idx: Index = "all",
        sample_idx: Index = "all",
        feature_idx: Union[Index, List[Index], Dict[str, Index]] = "all",
        as_df: bool = False,
    ):
        """Get imputed observations.

        Parameters
        ----------
        view_idx : Index, optional
            View index, by default "all"
        sample_idx : Index, optional
            Sample index, by default "all"
        feature_idx : Union[Index, List[Index], Dict[str, Index]], optional
            Feature index, by default "all"
        as_df : bool, optional
            Whether to return a pandas dataframe,
            by default False

        Returns
        -------
        dict
            Dictionary with view names as keys,
            and np.ndarray or pd.DataFrame as values.
        """

        obs = self.get_observations(view_idx, sample_idx, feature_idx, as_df=True)
        obs_hat = self.get_reconstructed(view_idx, sample_idx, feature_idx, as_df=True)
        obs_imputed = {vn: obs[vn].fillna(obs_hat[vn]) for vn in obs}

        if not as_df:
            obs_imputed = {
                vn: obs_imp.to_numpy() for vn, obs_imp in obs_imputed.items()
            }
        return obs_imputed

    def get_prior_masks(
        self,
        view_idx: Index = "all",
        factor_idx: Index = "all",
        feature_idx: Union[Index, List[Index], Dict[str, Index]] = "all",
        as_df: bool = False,
    ):
        """Get prior masks.

        Parameters
        ----------
        view_idx : Index, optional
            View index, by default "all"
        factor_idx : Index, optional
            Factor index, by default "all"
        feature_idx : Union[Index, List[Index], Dict[str, Index]], optional
            Feature index, by default "all"
        as_df : bool, optional
            Whether to return a pandas dataframe,
            by default False

        Returns
        -------
        dict
            Dictionary with view names as keys,
            and np.ndarray or pd.DataFrame as values.
        """

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
        view_idx: Index = "all",
        factor_idx: Index = "all",
        feature_idx: Union[Index, List[Index], Dict[str, Index]] = "all",
        as_df: bool = False,
    ):
        """Get factor loadings.

        Parameters
        ----------
        view_idx : Index, optional
            View index, by default "all"
        factor_idx : Index, optional
            Factor index, by default "all"
        feature_idx : Union[Index, List[Index], Dict[str, Index]], optional
            Feature index, by default "all"
        as_df : bool, optional
            Whether to return a pandas dataframe,
            by default False

        Returns
        -------
        dict
            Dictionary with view names as keys,
            and np.ndarray or pd.DataFrame as values.
        """
        self._raise_untrained_error()

        ws = self._guide.get_w(as_list=True)
        ws = [w * self._factor_signs.to_numpy()[:, np.newaxis] for w in ws]
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
        view_idx: Index = "all",
        cov_idx: Index = "all",
        feature_idx: Union[Index, List[Index], Dict[str, Index]] = "all",
        as_df: bool = False,
    ):
        """Get factor loadings.

        Parameters
        ----------
        view_idx : Index, optional
            View index, by default "all"
        cov_idx : Index, optional
            Covariate index, by default "all"
        feature_idx : Union[Index, List[Index], Dict[str, Index]], optional
            Feature index, by default "all"
        as_df : bool, optional
            Whether to return a pandas dataframe,
            by default False

        Returns
        -------
        dict
            Dictionary with view names as keys,
            and np.ndarray or pd.DataFrame as values.
        """
        self._raise_untrained_error()

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
        self, sample_idx: Index = "all", factor_idx: Index = "all", as_df: bool = False
    ):
        """Get factor scores.

        Parameters
        ----------
        sample_idx : Index, optional
            Sample index, by default "all"
        factor_idx : Index, optional
            Factor index, by default "all"
        as_df : bool, optional
            Whether to return a pandas dataframe,
            by default False

        Returns
        -------
        Union[np.ndarray, pd.DataFrame]
            A single np.ndarray or pd.DataFrame of shape `n_samples` x `n_factors`.
        """
        self._raise_untrained_error()

        return self._get_sample_attr(
            self._guide.get_z() * self._factor_signs.to_numpy(),
            sample_idx,
            other_idx=factor_idx,
            other_names=self.factor_names,
            as_df=as_df,
        )

    def get_covariates(
        self, sample_idx: Index = "all", cov_idx: Index = "all", as_df: bool = False
    ):
        """Get factor scores.

        Parameters
        ----------
        sample_idx : Index, optional
            Sample index, by default "all"
        cov_idx : Index, optional
            Covariate index, by default "all"
        as_df : bool, optional
            Whether to return a pandas dataframe,
            by default False

        Returns
        -------
        Union[np.ndarray, pd.DataFrame]
            A single np.ndarray or pd.DataFrame of shape `n_samples` x `n_covariates`.
        """
        return self._get_sample_attr(
            self.covariates,
            sample_idx,
            other_idx=cov_idx,
            other_names=self.covariate_names,
            as_df=as_df,
        )

    def _setup_model_guide(self, batch_size: int, scale_elbo: bool):
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
                reg_hs=self.reg_hs,
                nmf=[self.nmf[vn] for vn in self.view_names],
                scale_elbo=scale_elbo,
                device=self.device,
            )
            self._guide = MuVIGuide(self._model)
            self._built = True
        return self._built

    def _setup_optimizer(
        self, batch_size: int, n_epochs: int, learning_rate: float, optimizer: str
    ):
        """Setup SVI optimizer.

        Parameters
        ----------
        batch_size : int
            Batch size
        n_epochs : int
            Number of epochs, needed to schedule learning rate decay
        learning_rate : float
            Learning rate
        optimizer : str
            Optimizer as string, 'adam' or 'clipped'

        Returns
        -------
        pyro.optim.PyroOptim
            pyro or torch optimizer object
        """

        optim = Adam({"lr": learning_rate, "betas": (0.95, 0.999)})
        if optimizer.lower() == "clipped":
            n_iterations = int(n_epochs * (self.n_samples // batch_size))
            logger.info(f"Decaying learning rate over {n_iterations} iterations.")
            gamma = 0.1
            lrd = gamma ** (1 / n_iterations)
            optim = ClippedAdam({"lr": learning_rate, "lrd": lrd})

        self._optimizer = optim
        return self._optimizer

    def _setup_svi(
        self,
        optimizer: pyro.optim.PyroOptim,
        n_particles: int,
        scale: bool = True,
    ):
        """Setup stochastic variational inference.

        Parameters
        ----------
        optimizer : pyro.optim.PyroOptim
            pyro or torch optimizer
        n_particles : int
            Number of particles/samples used to form the ELBO (gradient) estimators
        scale : bool, optional
            Whether to scale ELBO by the number of samples, by default True

        Returns
        -------
        pyro.infer.SVI
            pyro SVI object
        """
        scaler = 1.0
        if scale:
            scaler = 1.0 / self.n_samples

        loss = pyro.infer.TraceMeanField_ELBO(
            retain_graph=True,
            num_particles=n_particles,
            vectorize_particles=True,
        )

        svi = pyro.infer.SVI(
            model=pyro.poutine.scale(self._model, scale=scaler),
            guide=pyro.poutine.scale(self._guide, scale=scaler),
            optim=optimizer,
            loss=loss,
        )

        self._svi = svi
        return self._svi

    def _setup_training_data(self):
        """Setup training components.
        Convert observations, covariates and prior scales to torch.Tensor and
        extract mask of missing values to mask SVI updates.

        Returns
        -------
        dict
            Dictionary of training data,
            same keys are expected as in the forward method of the MuVIModel
        """
        train_obs = torch.cat(
            [torch.Tensor(self.observations[vn]) for vn in self.view_names], 1
        )
        mask_obs = ~torch.isnan(train_obs)
        # replace all nans with zeros
        # self.presence mask takes care of gradient updates
        train_obs = torch.nan_to_num(train_obs)

        train_covs = None
        if self.covariates is not None:
            train_covs = torch.Tensor(self.covariates)

        train_prior_scales = None
        if self._informed:
            train_prior_scales = torch.cat(
                [torch.Tensor(self.prior_scales[vn]) for vn in self.view_names], 1
            )

        # make sure keys match the arguments in the forward method of the MuVIModel
        training_data = {
            "obs": train_obs,
            "mask": mask_obs,
            "covs": train_covs,
            "prior_scales": train_prior_scales,
        }
        return {k: v for k, v in training_data.items() if v is not None}

    def fit(
        self,
        batch_size: int = 0,
        n_epochs: int = 1000,
        n_particles: int = 0,
        learning_rate: float = 0.005,
        optimizer: str = "clipped",
        scale_elbo: bool = True,
        callbacks: Optional[List[Callable]] = None,
        verbose: bool = True,
        seed: Optional[int] = None,
    ):
        """Perform inference.

        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default 0 (all samples)
        n_epochs : int, optional
            Number of iterations over the whole dataset,
            by default 1000
        n_particles : int, optional
            Number of particles/samples used to form the ELBO (gradient) estimators,
            by default 0 (1000 // batch_size)
        learning_rate : float, optional
            Learning rate, by default 0.005
        scale_elbo : bool
            Whether to scale the ELBO across views, by default True
        optimizer : str, optional
            Optimizer as string, 'adam' or 'clipped', by default "clipped"
        callbacks : List[Callable], optional
            List of callbacks during training, by default None
        verbose : bool
            Whether to log progress, by default True
        seed : int, optional
            Training seed, by default None
        """

        # if invalid or out of bounds set to n_samples
        if batch_size is None or not (0 < batch_size <= self.n_samples):
            batch_size = self.n_samples

        if n_particles < 1:
            n_particles = max(1, 1000 // batch_size)
        if n_particles > 1:
            logger.info(f"Using {n_particles} particles in parallel.")
        logger.info("Preparing model and guide...")
        self._setup_model_guide(batch_size, scale_elbo)
        logger.info("Preparing optimizer...")
        opt = self._setup_optimizer(batch_size, n_epochs, learning_rate, optimizer)
        logger.info("Preparing SVI...")
        svi = self._setup_svi(opt, n_particles, scale=True)
        logger.info("Preparing training data...")
        training_data = self._setup_training_data()
        training_prior_scales = training_data.pop("prior_scales", None)
        if training_prior_scales is not None:
            training_prior_scales = training_prior_scales.to(self.device)

        if batch_size < self.n_samples:
            logger.info(f"Using batches of size `{batch_size}`.")

            data_loader = DataLoader(
                DictDataset(training_data, idx_key="sample_idx"),
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=str(self.device) != "cpu",
                drop_last=False,
            )

            def _step():
                iteration_loss = 0
                for _, tensor_dict in enumerate(data_loader):
                    iteration_loss += svi.step(
                        **{k: v.to(self.device) for k, v in tensor_dict.items()},
                        prior_scales=training_prior_scales,
                    )
                return iteration_loss

        else:
            logger.info("Using complete dataset.")
            # move all data to device once
            training_data = {k: v.to(self.device) for k, v in training_data.items()}

            def _step():
                return svi.step(
                    None,
                    **training_data,
                    prior_scales=training_prior_scales,
                )

        if seed is not None:
            try:
                seed = int(seed)
            except ValueError:
                logger.warning(f"Could not convert `{seed}` to integer.")
                seed = None

        if seed is None:
            seed = int(time.strftime("%y%m%d%H%M"))

        logger.info(f"Setting training seed to `{seed}`.")
        pyro.set_rng_seed(seed)
        # clean start
        logger.info("Cleaning parameter store.")
        pyro.enable_validation(True)
        pyro.clear_param_store()

        logger.info("Starting training...")
        self._trained = True
        stop_early = False
        history = []
        pbar = range(n_epochs)
        if verbose:
            pbar = tqdm(pbar)
            window_size = 5

        try:
            for epoch_idx in pbar:
                epoch_loss = _step()
                history.append(epoch_loss)
                if verbose and (
                    epoch_idx % window_size == 0 or epoch_idx == n_epochs - 1
                ):
                    pbar.set_postfix({"ELBO": epoch_loss})
                if callbacks is not None:
                    # TODO: dont really like this, a bit sloppy
                    stop_early = any(callback(history) for callback in callbacks)
                    if stop_early:
                        break
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, stopping training and saving progress...")

        self._training_log = {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "n_particles": n_particles,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "verbose": verbose,
            "seed": seed,
            "history": history,
            "n_iter": len(history),
            "stop_early": stop_early,
        }
        logger.info("Call `model._training_log` to inspect the training progress.")
        # reset cache in case it was initialized by any of the callbacks
        self._cache = None


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
        reg_hs: bool = True,
        nmf: Optional[List[bool]] = None,
        scale_elbo: bool = True,
        device=None,
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
            either "normal" or "bernoulli", by default None
        global_prior_scale : float, optional
            Determine the level of global sparsity, by default 1.0
        reg_hs : bool, optional
            Whether to use the regularized version of HS,
            by default True
        nmf : List[bool], optional
            Whether to use non-negative matrix factorization,
            by default empty (False for all views)
        scale_elbo : bool
            Whether to scale the ELBO across views, by default True
        device : str, optional
            Device to run computations on, by default "cuda" (GPU)
        """
        super().__init__(name="MuVIModel")
        self.n_samples = n_samples
        self.n_subsamples = n_subsamples
        self.n_features = n_features
        self.feature_offsets = [0, *np.cumsum(self.n_features).tolist()]
        self.n_views = len(self.n_features)
        self.n_factors = n_factors
        self.n_covariates = n_covariates
        self.likelihoods = likelihoods
        self.same_likelihood = len(set(self.likelihoods)) == 1
        self.global_prior_scale = (
            1.0 if global_prior_scale is None else global_prior_scale
        )
        self.reg_hs = reg_hs
        if nmf is None:
            nmf = [False for _ in range(self.n_views)]
        self.nmf = nmf
        # only needed if nmf is True
        self.pos_transform = torch.nn.ReLU()

        self.scale_elbo = scale_elbo
        self.view_scales = np.ones(self.n_views)
        if self.scale_elbo and self.n_views > 1:
            self.view_scales = (self.n_views / (self.n_views - 1)) * (
                1.0 - np.array([nf / sum(n_features) for nf in n_features])
            )
        self.device = device

    def get_plate(self, name: str, **kwargs):
        """Get the sampling plate.

        Parameters
        ----------
        name : str
            Name of the plate

        Returns
        -------
        PlateMessenger
            A pyro plate.
        """
        plate_kwargs = {
            "view": {"name": "view", "size": self.n_views, "dim": -1},
            "factor_left": {"name": "factor_left", "size": self.n_factors, "dim": -2},
            "sample": {"name": "sample", "size": self.n_samples, "dim": -2},
            "covariate": {"name": "covariate", "size": self.n_covariates, "dim": -2},
        }
        for m, n_features in zip(range(self.n_views), self.n_features):
            plate_kwargs[f"feature_{m}"] = {
                "name": f"feature_{m}",
                "size": n_features,
                "dim": -1,
            }
        return pyro.plate(device=self.device, **{**plate_kwargs[name], **kwargs})

    def _zeros(self, size):
        return torch.zeros(size, device=self.device)

    def _ones(self, size):
        return torch.ones(size, device=self.device)

    def forward(
        self,
        sample_idx: torch.Tensor,
        obs: torch.Tensor,
        mask: torch.Tensor,
        covs: Optional[torch.Tensor] = None,
        prior_scales: Optional[torch.Tensor] = None,
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

        view_plate = self.get_plate("view")
        factor_l_plate = self.get_plate("factor_left")
        feature_plates = [self.get_plate(f"feature_{m}") for m in range(self.n_views)]
        sample_plate = self.get_plate("sample", subsample=sample_idx)

        if self.n_covariates > 0:
            covariate_plate = self.get_plate("covariate")

        with view_plate:
            output_dict["view_scale"] = pyro.sample(
                "view_scale", dist.HalfCauchy(self._ones((1,)))
            )
            with factor_l_plate:
                output_dict["factor_scale"] = pyro.sample(
                    "factor_scale",
                    dist.HalfCauchy(self._ones((1,))),
                )

        for m in range(self.n_views):
            with feature_plates[m]:
                with factor_l_plate:
                    output_dict[f"local_scale_{m}"] = pyro.sample(
                        f"local_scale_{m}",
                        dist.HalfCauchy(self._ones((1,))),
                    )

                    w_scale = (
                        output_dict[f"local_scale_{m}"]
                        * output_dict["factor_scale"][..., m : m + 1]
                        * output_dict["view_scale"][..., m : m + 1]
                    )

                    if self.reg_hs:
                        output_dict[f"caux_{m}"] = pyro.sample(
                            f"caux_{m}",
                            dist.InverseGamma(
                                0.5 * self._ones((1,)), 0.5 * self._ones((1,))
                            ),
                        )
                        c = torch.sqrt(output_dict[f"caux_{m}"])
                        if prior_scales is not None:
                            c = (
                                c
                                * prior_scales[
                                    :,
                                    self.feature_offsets[m] : self.feature_offsets[
                                        m + 1
                                    ],
                                ]
                            )
                        w_scale = (self.global_prior_scale * c * w_scale) / torch.sqrt(
                            c**2 + w_scale**2
                        )

                    output_dict[f"w_{m}"] = pyro.sample(
                        f"w_{m}",
                        dist.Normal(
                            self._zeros((1,)),
                            w_scale,
                        ),
                    )

                    if self.nmf[m]:
                        output_dict[f"w_{m}"] = self.pos_transform(
                            output_dict[f"w_{m}"]
                        )

                if self.n_covariates > 0:
                    with covariate_plate:
                        output_dict[f"beta_{m}"] = pyro.sample(
                            f"beta_{m}",
                            dist.Normal(self._zeros((1,)), self._ones((1,))),
                        )

                output_dict[f"sigma_{m}"] = pyro.sample(
                    f"sigma_{m}",
                    dist.InverseGamma(self._ones((1,)), self._ones((1,))),
                )

        with sample_plate:
            output_dict["z"] = pyro.sample(
                "z",
                dist.Normal(
                    self._zeros((self.n_factors,)),
                    self._ones((self.n_factors,)),
                ),
            )
            if any(self.nmf):
                output_dict["z"] = self.pos_transform(output_dict["z"])
            for m in range(self.n_views):
                with feature_plates[m]:
                    y_loc = torch.matmul(output_dict["z"], output_dict[f"w_{m}"])
                    if self.n_covariates > 0:
                        y_loc = y_loc + torch.matmul(covs, output_dict[f"beta_{m}"])

                    if self.likelihoods[m] == "normal":
                        y_dist = dist.Normal(
                            y_loc,
                            torch.sqrt(output_dict[f"sigma_{m}"]),
                        )
                    else:
                        y_dist = dist.Bernoulli(logits=y_loc)

                    feature_idx = slice(
                        self.feature_offsets[m],
                        self.feature_offsets[m + 1],
                    )

                    with pyro.poutine.mask(
                        mask=mask[..., feature_idx]
                    ), pyro.poutine.scale(scale=self.view_scales[m]):
                        output_dict[f"y_{m}"] = pyro.sample(
                            f"y_{m}",
                            y_dist,
                            obs=obs[..., feature_idx],
                            infer={"is_auxiliary": True},
                        )

        return output_dict


class MuVIGuide(PyroModule):
    def __init__(
        self,
        model,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
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

        self.site_to_dist = self.setup()

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
        n_views = self.model.n_views
        n_samples = self.model.n_samples
        n_factors = self.model.n_factors
        n_features = self.model.n_features
        n_covariates = self.model.n_covariates

        site_to_shape = {
            "z": (n_samples, n_factors),
            "view_scale": (n_views,),
            "factor_scale": (n_factors, n_views),
        }

        normal_sites = ["z"]
        for m in range(n_views):
            site_to_shape[f"local_scale_{m}"] = (n_factors, n_features[m])
            site_to_shape[f"caux_{m}"] = (n_factors, n_features[m])
            site_to_shape[f"w_{m}"] = (n_factors, n_features[m])
            site_to_shape[f"beta_{m}"] = (n_covariates, n_features[m])
            site_to_shape[f"sigma_{m}"] = (n_features[m],)

            normal_sites.append(f"w_{m}")
            normal_sites.append(f"beta_{m}")

        site_to_dist = {
            k: "Normal" if k in normal_sites else "LogNormal" for k in site_to_shape
        }

        for name, shape in site_to_shape.items():
            deep_setattr(
                self.locs,
                name,
                PyroParam(
                    self.init_loc * self.model._ones(shape),
                    constraints.real,
                ),
            )

            deep_setattr(
                self.scales,
                name,
                PyroParam(
                    self.init_scale * self.model._ones(shape),
                    constraints.softplus_positive,
                ),
            )

        return site_to_dist

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
        if self.site_to_dist[name] == "LogNormal":
            mode = (loc - scale.square()).exp()
        if (name == "z") and any(self.model.nmf):
            mode = self.model.pos_transform(mode)
        for m in range(self.model.n_views):
            if name == f"w_{m}" and self.model.nmf[m]:
                mode = self.model.pos_transform(mode)
        return mode.clone()

    @torch.no_grad()
    def _get_map_estimate(self, param_name: str, is_list_param: bool, as_list: bool):
        is_list_param |= as_list
        if is_list_param:
            params = [self.mode(f"{param_name}_{m}") for m in range(self.model.n_views)]
        else:
            params = [self.mode(param_name)]

        vector_params = ["view_scale", "sigma"]

        if param_name in vector_params:
            params = [param[None, :] for param in params]
        params = [param.cpu().detach().numpy() for param in params]

        if as_list:
            return params
        return np.concatenate(params, axis=1)

    def get_sigma(self, as_list: bool = False):
        """Get the marginal feature scales."""
        return self._get_map_estimate("sigma", True, as_list=as_list)

    def get_view_scale(self):
        """Get the view scales."""
        return self._get_map_estimate("view_scale", False, False)

    def get_factor_scale(self):
        """Get the factor scales."""
        return self._get_map_estimate("factor_scale", False, False).T

    def get_local_scale(self, as_list: bool = False):
        """Get the local scales."""
        return self._get_map_estimate("local_scale", True, as_list=as_list)

    def get_caux(self, as_list: bool = False):
        """Get the c auxiliaries."""
        return self._get_map_estimate("caux", True, as_list=as_list)

    def get_z(self):
        """Get the factor scores."""
        return self._get_map_estimate("z", False, False)

    def get_w(self, as_list: bool = False):
        """Get the factor loadings."""
        return self._get_map_estimate("w", True, as_list=as_list)

    def get_beta(self, as_list: bool = False):
        """Get the beta coefficients."""
        return self._get_map_estimate("beta", True, as_list=as_list)

    def _sample(self, name, index=None):
        loc, scale = self._get_loc_and_scale(name)
        if index is not None:
            loc = loc.index_select(0, index)
            scale = scale.index_select(0, index)
        if self.site_to_dist[name] == "LogNormal":
            return pyro.sample(name, dist.LogNormal(loc, scale))
        if self.site_to_dist[name] == "Dirichlet":
            return pyro.sample(name, dist.Dirichlet(scale))
        return pyro.sample(name, dist.Normal(loc, scale))

    def forward(
        self,
        sample_idx: torch.Tensor,
        obs: torch.Tensor,
        mask: torch.Tensor,
        covs: Optional[torch.Tensor] = None,
        prior_scales: Optional[torch.Tensor] = None,
    ):
        """Approximate posterior."""
        output_dict = {}

        view_plate = self.model.get_plate("view")
        factor_l_plate = self.model.get_plate("factor_left")
        feature_plates = [
            self.model.get_plate(f"feature_{m}") for m in range(self.model.n_views)
        ]
        sample_plate = self.model.get_plate("sample", subsample=sample_idx)

        if self.model.n_covariates > 0:
            covariate_plate = self.model.get_plate("covariate")

        with view_plate:
            output_dict["view_scale"] = self._sample("view_scale")
            with factor_l_plate:
                output_dict["factor_scale"] = self._sample("factor_scale")

        for m in range(self.model.n_views):
            with feature_plates[m]:
                with factor_l_plate:
                    output_dict[f"local_scale_{m}"] = self._sample(f"local_scale_{m}")
                    if self.model.reg_hs:
                        output_dict[f"caux_{m}"] = self._sample(f"caux_{m}")
                    output_dict[f"w_{m}"] = self._sample(f"w_{m}")

                if self.model.n_covariates > 0:
                    with covariate_plate:
                        output_dict[f"beta_{m}"] = self._sample(f"beta_{m}")

                output_dict[f"sigma_{m}"] = self._sample(f"sigma_{m}")

        with sample_plate as indices:
            output_dict["z"] = self._sample("z", indices)
        return output_dict
