import logging
import os
import pickle
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .index import _normalize_index

logger = logging.getLogger(__name__)

Index = Union[int, str, List[int], List[str], np.ndarray, pd.Index]
SingleView = Union[np.ndarray, pd.DataFrame]
MultiView = Union[Dict[str, SingleView], List[SingleView]]
MultiGroup = MultiView
MultiViewGroup = Union[Dict[str, MultiView], List[MultiView]]


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return index, tuple(tensor[index] for tensor in self.tensors)


class MuVI(PyroModule):
    def __init__(
        self,
        observations: MultiViewGroup,
        prior_masks: MultiView = None,
        covariates: MultiGroup = None,
        prior_confidence: float = 0.99,
        n_factors: int = None,
        group_names: List[str] = None,
        view_names: List[str] = None,
        likelihoods: Union[Dict[str, str], List[str]] = None,
        use_gpu: bool = True,
    ):
        """MuVI module.

        Parameters
        ----------
        observations : MultiViewGroup
            Collection of observations as list (group) of lists (view) or dict of dicts.
        prior_masks : MultiView, optional
            Prior feature sets as a collection of binary masks,
            by default None
        covariates : MultiGroup, optional
            Additional observed covariates across all groups, by default None
        prior_confidence : float, optional
            Confidence of prior belief from 0 to 1 (exclusive),
            typical values are between 0.95 and 0.999,
            by default 0.99
        n_factors : int, optional
            Number of latent factors,
            can be omitted when providing prior masks,
            by default None
        group_names : List[str], optional
            List of names for each group,
            determines group order as well,
            by default None
        view_names : List[str], optional
            List of names for each view,
            determines view order as well,
            by default None
        likelihoods : Union[Dict[str, str], List[str]], optional
            Likelihoods for each view,
            either "normal" or "bernoulli",
            by default None (all "normal")
        use_gpu : bool, optional
            Whether to train on a GPU, by default True
        """
        super().__init__(name="MuVI")
        self.observations = self._setup_observations(
            observations, group_names, view_names
        )
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

        self._model = None
        self._guide = None
        self._informed = self.prior_masks is not None
        self._built = False
        self._trained = False
        self._cache = None

    @property
    def _factor_signs(self):
        signs = 1.0
        if self._trained:
            w = self._guide.get_w()
            # TODO: looking at top loadings works better than including all
            # 100 feels a bit arbitrary though
            w = np.array(
                list(map(lambda x, y: y[x], np.argsort(-np.abs(w), axis=1)[:, :100], w))
            )
            signs = (w.sum(axis=1) > 0) * 2 - 1

        return pd.Series(signs, index=self.factor_names, dtype=np.float32)

    def _validate_index(self, idx):
        if not is_string_dtype(idx):
            logger.warning("Transforming to str index.")
            return idx.astype(str)
        return idx

    def _setup_groups(self, observations, group_names):
        n_groups = len(observations)
        if n_groups == 1:
            logger.warning("Running MuVI on a single group.")

        if group_names is None:
            logger.warning("No group names provided!")
            if isinstance(observations, list):
                logger.info(
                    "Setting the name of each group to `group_idx` "
                    "for list observations."
                )
                group_names = [f"group_{g}" for g in range(n_groups)]
            if isinstance(observations, dict):
                logger.info(
                    "Setting the group names to the sorted list "
                    "of dictonary keys in observations."
                )
                group_names = sorted(observations.keys())

        if n_groups > len(group_names):
            logger.warning(
                "Number of groups is larger than the length of `group_names`, "
                "using only the subset of groups defined in `group_names`."
            )
            n_groups = len(group_names)

        group_names = pd.Index(group_names)
        # if list convert to dict
        if isinstance(observations, list):
            observations = dict(zip(group_names, observations))

        self.n_groups = n_groups
        self.group_names = self._validate_index(group_names)
        return observations

    def _setup_views(self, observations, view_names):
        # observations now a dictionary of lists/dicts
        # first check whether all groups have the same number of views
        n_views = None
        for gn in self.group_names:
            group_n_views = len(observations[gn])
            if n_views is None:
                n_views = group_n_views
            else:
                if n_views != group_n_views:
                    raise ValueError(
                        f"Group `{gn}` has {group_n_views} views "
                        f"instead of {n_views}, "
                        "all groups must have the same number of views."
                    )

        if n_views == 1:
            logger.warning("Running MuVI on a single view.")

        for gn in self.group_names:
            # TODO: does not consider mixture of list and dict
            # for the view collection (does not strictly check view names..)
            if view_names is None:
                logger.warning("No view names provided!")
                if isinstance(observations[gn], list):
                    logger.info(
                        "Setting the name of each view to `view_idx` "
                        f"for list observations in group `{gn}`."
                    )
                    view_names = [f"view_{m}" for m in range(n_views)]
                if isinstance(observations[gn], dict):
                    logger.info(
                        "Setting the view names to the sorted list "
                        f"of dictonary keys for observations in group `{gn}`."
                    )
                    view_names = sorted(observations[gn].keys())

            if n_views > len(view_names):
                logger.warning(
                    "Number of views is larger than the length of `view_names`, "
                    "using only the subset of views defined in `view_names`."
                )
                n_views = len(view_names)

            view_names = pd.Index(view_names)
            # if list convert to dict
            if isinstance(observations[gn], list):
                observations[gn] = dict(zip(view_names, observations[gn]))

        self.n_views = n_views
        self.view_names = self._validate_index(view_names)
        return observations

    def _setup_samples(self, observations):

        # samples expected to be different for groups and identical for views
        n_samples = {gn: 0 for gn in self.group_names}
        sample_names = {gn: None for gn in self.group_names}

        for gn in self.group_names:
            for vn in self.view_names:
                view_n_samples = observations[gn][vn].shape[0]
                if n_samples[gn] == 0:
                    n_samples[gn] = view_n_samples
                if n_samples[gn] != view_n_samples:
                    raise ValueError(
                        f"View `{vn}` in group `{gn}` has {view_n_samples} samples "
                        f"instead of {n_samples[gn]}, "
                        "all views must have the same number of samples."
                    )

                if isinstance(observations[gn][vn], pd.DataFrame):
                    logger.info("pd.DataFrame detected.")

                    view_sample_names = observations[gn][vn].index.copy()

                    if sample_names[gn] is None:
                        logger.info(
                            f"Storing the index of the view `{vn}` "
                            f"in group `{gn}` as sample names."
                        )
                        sample_names[gn] = view_sample_names

                    if any(sample_names[gn] != view_sample_names):
                        logger.info(
                            f"Sample names for view `{vn}` in group `{gn}` do not "
                            f"match the sample names of view `{self.view_names[0]}` "
                            f"within the same group, sorting names "
                            f"according to the first view `{self.view_names[0]}`."
                        )
                        observations[gn][vn] = observations[gn][vn].loc[
                            sample_names[self.group_names[0]], :
                        ]

            if sample_names[gn] is None:
                logger.info(
                    f"Setting the name of each sample in `{gn}` to `{gn}_sample_idx`."
                )
                sample_names[gn] = pd.Index(
                    [f"{gn}_sample_{i}" for i in range(n_samples[gn])]
                )

            sample_names[gn] = self._validate_index(sample_names[gn])

        self.n_samples = n_samples
        self.n_all_samples = sum(n_samples.values())
        self.sample_names = sample_names

        return observations

    def _setup_features(self, observations):

        # features expected to be different for views and identical for groups
        n_features = {vn: 0 for vn in self.view_names}
        feature_names = {vn: None for vn in self.view_names}

        for vn in self.view_names:
            for gn in self.group_names:
                group_n_features = observations[gn][vn].shape[1]
                if n_features[vn] == 0:
                    n_features[vn] = group_n_features
                if n_features[vn] != group_n_features:
                    raise ValueError(
                        f"View `{vn}` in group `{gn}` has {group_n_features} features "
                        f"instead of {n_features[vn]}, "
                        "all groups must have the same number of features."
                    )

                if isinstance(observations[gn][vn], pd.DataFrame):
                    logger.info("pd.DataFrame detected.")

                    group_feature_names = observations[gn][vn].columns.copy()

                    if feature_names[vn] is None:
                        logger.info(
                            f"Storing the columns of the view `{vn}` "
                            f"in group `{gn}` as feature names."
                        )
                        feature_names[vn] = group_feature_names

                    if any(feature_names[vn] != group_feature_names):
                        logger.info(
                            f"Feature names for view `{vn}` in group `{gn}` do not "
                            f"match the feature names of group `{self.group_names[0]}` "
                            f"within the same view, sorting names "
                            f"according to the first group `{self.group_names[0]}`."
                        )
                        observations[gn][vn] = observations[gn][vn].loc[
                            :, feature_names[self.view_names[0]]
                        ]

            if feature_names[vn] is None:
                logger.info(
                    f"Setting the name of each feature in `{vn}` to `{vn}_feature_idx`."
                )
                feature_names[vn] = pd.Index(
                    [f"{vn}_feature_{j}" for j in range(n_features[vn])]
                )

            feature_names[vn] = self._validate_index(feature_names[vn])

        self.n_features = n_features
        self.n_all_features = sum(n_features.values())
        self.feature_names = feature_names

        return observations

    def _setup_observations(self, observations, group_names, view_names):
        if observations is None:
            raise ValueError(
                "Observations is None, please provide a valid list of observations."
            )

        observations = self._setup_groups(observations, group_names)
        observations = self._setup_views(observations, view_names)
        # from now on only working with nested dictonaries
        observations = self._setup_samples(observations)
        observations = self._setup_features(observations)

        # keep only numpy arrays, convert to np.float32 dtypes
        return {
            gn: {
                vn: (
                    obs.to_numpy(dtype=np.float32)
                    if isinstance(obs, pd.DataFrame)
                    else np.array(obs, dtype=np.float32)
                )
                for vn, obs in group_obs.items()
            }
            for gn, group_obs in observations.items()
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
            masks = {self.view_names[m]: mask for m, mask in enumerate(masks)}

        informed_views = [vn for vn in self.view_names if vn in masks]

        n_prior_factors = masks[informed_views[0]].shape[0]
        if n_factors is None:
            n_factors = n_prior_factors

        if n_prior_factors > n_factors:
            logger.warning(
                "Prior mask informs more factors than the pre-defined `n_factors`. "
                "Updating `n_factors` to %s.",
                n_prior_factors,
            )
            n_factors = n_prior_factors

        n_dense_factors = 0
        if n_prior_factors < n_factors:
            logger.warning(
                "Prior mask informs fewer factors than the pre-defined `n_factors`. "
                "Informing only the first %s factors, the rest remains uninformed.",
                n_prior_factors,
            )
            # extend all prior masks with additional uninformed factors
            n_dense_factors = n_factors - n_prior_factors

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
            if view_mask.shape[0] != n_factors - n_dense_factors:
                raise ValueError(
                    f"Mask `{vn}` has {view_mask.shape[0]} factors "
                    f"instead of {n_factors - n_dense_factors}, "
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
                        "Storing the index of the mask `%s` as factor names.", vn
                    )
                    factor_names = mask_factor_names
                if any(factor_names != mask_factor_names):
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
            factor_names = [f"factor_{k}" for k in range(n_factors)]
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
                    [vm, np.ones((n_dense_factors, self.n_features[vn]))]
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
        logger.info("Likelihoods set to %s", likelihoods)
        return likelihoods

    def _setup_covariates(self, covariates):

        n_covariates = 0
        covariate_names = None
        if covariates is None:
            self.n_covariates = n_covariates
            self.covariate_names = covariate_names
            return None

        if isinstance(covariates, list):
            covariates = {
                self.group_names[g]: group_covs
                for g, group_covs in enumerate(covariates)
            }
        n_covariates = {gn: covariates[gn].shape[1] for gn in self.group_names}
        n_covariates_set = set(n_covariates.values())
        if len(n_covariates_set) != 1:
            raise NotImplementedError(
                "Handling variable number of covariates across groups is not supported."
            )
        n_covariates = n_covariates_set.pop()

        for gn in self.group_names:
            if isinstance(covariates[gn], pd.DataFrame):
                logger.info("pd.DataFrame detected.")
                if covariates[gn].shape[0] != self.n_samples[gn]:
                    raise ValueError(
                        f"Number of the covariate samples ({covariates[gn].shape[0]}) "
                        f"in group `{gn}` does not match the number of "
                        f"observed samples ({self.n_samples[gn]}) "
                        f"within the same group."
                    )

                if any(covariates[gn].index != self.sample_names[gn]):
                    logger.info(
                        f"Sample names for the covariates in group `{gn}` "
                        "do not match the sample names of the observations, "
                        "sorting names according to the observations."
                    )
                    covariates[gn] = covariates[gn].loc[self.sample_names[gn], :]

                if covariate_names is None:
                    covariate_names = covariates[gn].columns.copy()
                if any(covariates[gn].columns != covariate_names):
                    logger.info(
                        f"Covariate names for the covariates in group `{gn}` "
                        f"do not match the covariate names of group `{self.group_names[0]}`, "
                        f"sorting names according to the the names in `{self.group_names[0]}`."
                    )

                covariates[gn] = covariates[gn].to_numpy()
            if covariate_names is None:
                covariate_names = pd.Index(
                    [f"covariate_{k}" for k in range(n_covariates)]
                )

        self.n_covariates = n_covariates
        self.covariate_names = covariate_names

        # keep only numpy arrays, convert to np.float32 dtypes
        return {
            gn: (
                cov.to_numpy(dtype=np.float32)
                if isinstance(cov, pd.DataFrame)
                else np.array(cov, dtype=np.float32)
            )
            for gn, cov in covariates.items()
        }

    def _raise_untrained_error(self):
        if not self._trained:
            raise AttributeError(
                "Requested attribute cannot be found on an untrained model! "
                "Run the `fit` method to train a MuVI model."
            )
        return True

    def _get_attr(
        self,
        attr,
        outer_idx,
        outer_names,
        inner_idx,
        inner_names,
        other_idx,
        other_names,
        as_df,
    ):
        if outer_idx is None and isinstance(inner_idx, dict):
            outer_idx = []
            for key in inner_idx.keys():
                oi = key
                if isinstance(oi, int):
                    oi = outer_names[oi]
                outer_idx.append(oi)
        if outer_idx is None:
            raise IndexError(
                "Invalid indices, `outer_idx` is None "
                "and `inner_idx` is not a dictionary."
            )
        # get relevant outer names
        outer_names = _normalize_index(outer_idx, outer_names, as_idx=False)
        if isinstance(inner_idx, (str, int)):
            if inner_idx == "all":
                inner_idx = ["all" for _ in range(len(outer_names))]
            else:
                inner_idx = [inner_idx]

        # convert to list if any list-like indexing
        if isinstance(inner_idx, (list, np.ndarray, pd.Index)):
            inner_idx = list(inner_idx)

        # check if valid combination of indices
        if len(outer_names) == 1 and len(inner_idx) != 1:
            if isinstance(inner_idx[0], list):
                logger.warning(
                    "`inner_idx` suggests indices for more than one outer dim, "
                    "keeping only the first list of indices."
                )
                inner_idx = inner_idx[0]
            inner_idx = [inner_idx]

        if len(outer_names) != len(inner_idx):
            logger.warning(
                "`outer_idx` does not match the keys of `inner_idx`, "
                "`outer_idx` has precedence over `inner_idx`."
            )

        if isinstance(inner_idx, list):
            inner_idx = {on: inner_idx[i] for i, on in enumerate(outer_names)}

        if isinstance(inner_idx, dict):
            inner_idx = {
                # TODO!: (self.view_names[k] if isinstance(k, int) else k): v
                (outer_names[k] if isinstance(k, int) else k): v
                for k, v in inner_idx.items()
            }

        # normalise dictionary
        inner_idx = {
            on: _normalize_index(inner_idx[on], inner_names[on]) for on in outer_names
        }
        other_idx = _normalize_index(other_idx, other_names)

        attr = {on: attr[on][other_idx, :][:, inner_idx[on]] for on in outer_names}
        if as_df:
            attr = {
                on: pd.DataFrame(
                    attr[on],
                    index=other_names[other_idx],
                    columns=inner_names[on][inner_idx[on]],
                )
                for on in outer_names
            }
        return attr

    def _get_view_attr(
        self,
        attr,
        view_idx,
        feature_idx,
        other_idx,
        other_names,
        as_df,
    ):

        return self._get_attr(
            attr,
            outer_idx=view_idx,
            outer_names=self.view_names,
            inner_idx=feature_idx,
            inner_names=self.feature_names,
            other_idx=other_idx,
            other_names=other_names,
            as_df=as_df,
        )

    def _get_group_attr(
        self,
        attr,
        group_idx,
        sample_idx,
        other_idx,
        other_names,
        as_df,
    ):

        return self._get_attr(
            attr,
            outer_idx=group_idx,
            outer_names=self.group_names,
            inner_idx=sample_idx,
            inner_names=self.sample_names,
            other_idx=other_idx,
            other_names=other_names,
            as_df=as_df,
        )

    def get_observations(
        self,
        group_idx: Index = "all",
        view_idx: Index = "all",
        sample_idx: Index = "all",
        feature_idx: Union[Index, List[Index], Dict[str, Index]] = "all",
        as_df: bool = False,
        predicted: bool = False,
    ):
        """Get observations.

        Parameters
        ----------
        group_idx : Index, optional
            Group index, by default "all"
        view_idx : Index, optional
            View index, by default "all"
        sample_idx : Index, optional
            Sample index, by default "all"
        feature_idx : Union[Index, List[Index], Dict[str, Index]], optional
            Feature index, by default "all"
        as_df : bool, optional
            Whether to return a pandas dataframe,
            by default False
        predicted : bool, optional
            Whether to return a the predicted observations,
            i.e. Z @ W, by default False

        Returns
        -------
        dict
            Dictionary with view names as keys,
            and np.ndarray or pd.DataFrame as values.
        """
        # TODO: WIP, does not consider nested structures (dict/list of dicts/lists)
        group_names = _normalize_index(group_idx, self.group_names, as_idx=False)

        obs = {}

        for gn in group_names:
            obs[gn] = self.observations[gn]
            if predicted:
                obs[gn] = {
                    vn: self.get_factor_scores(gn)[gn] @ loadings
                    for vn, loadings in self.get_factor_loadings().items()
                }
            obs[gn] = self._get_view_attr(
                obs[gn],
                view_idx,
                feature_idx,
                other_idx=sample_idx,
                other_names=self.sample_names,
                as_df=as_df,
            )

        return obs

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
        self,
        group_idx: Index = "all",
        sample_idx: Index = "all",
        factor_idx: Index = "all",
        as_df: bool = False,
    ):
        """Get factor scores.

        Parameters
        ----------
        group_idx : Index, optional
            Group index, by default "all"
        sample_idx : Index, optional
            Sample index, by default "all"
        factor_idx : Index, optional
            Factor index, by default "all"
        as_df : bool, optional
            Whether to return a pandas dataframe,
            by default False

        Returns
        -------
        dict
            Dictionary with group names as keys,
            and np.ndarray or pd.DataFrame as values.
        """
        self._raise_untrained_error()

        zs = self._guide.get_z(as_list=True)
        zs = [
            zs[g] * self._factor_signs[gn].to_numpy()[:, np.newaxis]
            for g, gn in enumerate(self.group_names)
        ]
        return self._get_group_attr(
            {gn: zs[g] for g, gn in enumerate(self.group_names)},
            group_idx,
            sample_idx,
            other_idx=factor_idx,
            other_names=self.factor_names,
            as_df=as_df,
        )

    def get_covariates(
        self,
        group_idx: Index = "all",
        sample_idx: Index = "all",
        cov_idx: Index = "all",
        as_df: bool = False,
    ):
        """Get factor scores.

        Parameters
        ----------
        group_idx : Index, optional
            Group index, by default "all"
        sample_idx : Index, optional
            Sample index, by default "all"
        cov_idx : Index, optional
            Covariate index, by default "all"
        as_df : bool, optional
            Whether to return a pandas dataframe,
            by default False

        Returns
        -------
        dict
            Dictionary with group names as keys,
            and np.ndarray or pd.DataFrame as values.
        """
        return self._get_group_attr(
            self.covariates,
            group_idx,
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
                n_samples=[self.n_samples[gn] for gn in self.group_names],
                n_subsamples=batch_size,
                n_features=[self.n_features[vn] for vn in self.view_names],
                n_factors=self.n_factors,
                n_covariates=self.n_covariates,
                likelihoods=[self.likelihoods[vn] for vn in self.view_names],
                device=self.device,
            )
            self._guide = MuVIGuide(self._model, device=self.device)
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
            n_iterations = int(n_epochs * (self.n_all_samples // batch_size))
            logger.info("Decaying learning rate over %s iterations.", n_iterations)
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
            scaler = 1.0 / self.n_all_samples

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
                torch.cat(
                    [torch.Tensor(self.observations[gn][vn]) for vn in self.view_names],
                    1,
                )
                for gn in self.group_names
            ],
            0,
        )
        mask_obs = ~torch.isnan(train_obs)
        # replace all nans with zeros
        # self.presence mask takes care of gradient updates
        train_obs = torch.nan_to_num(train_obs)

        train_covs = None
        if self.covariates is not None:
            train_covs = torch.cat(
                [torch.Tensor(self.covariates[gn]) for gn in self.group_names], 0
            )

        train_prior_scales = None
        if self._informed:
            train_prior_scales = torch.cat(
                [torch.Tensor(self.prior_scales[vn]) for vn in self.view_names], 1
            )

        return train_obs, mask_obs, train_covs, train_prior_scales

    def fit(
        self,
        batch_size: int = 0,
        n_epochs: int = 1000,
        n_particles: int = None,
        learning_rate: float = 0.005,
        optimizer: str = "clipped",
        callbacks: List[Callable] = None,
        verbose: bool = True,
        seed: str = None,
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
            by default 1000 // batch_size
        learning_rate : float, optional
            Learning rate, by default 0.005
        optimizer : str, optional
            Optimizer as string, 'adam' or 'clipped', by default "clipped"
        callbacks : List[Callable], optional
            List of callbacks during training, by default None
        verbose : bool, optional
            Whether to log progress, by default 1
        seed : str, optional
            Training seed, by default None

        Returns
        -------
        tuple
            Tuple of (elbo history, whether training stopped early)
        """

        # if invalid or out of bounds set to n_samples
        if batch_size is None or not (0 < batch_size <= self.n_all_samples):
            batch_size = self.n_all_samples

        if n_particles is None:
            n_particles = max(1, 1000 // batch_size)
        logger.info("Using %s particles on parallel", n_particles)
        logger.info("Preparing model and guide...")
        self._setup_model_guide(batch_size)
        logger.info("Preparing optimizer...")
        optimizer = self._setup_optimizer(
            batch_size, n_epochs, learning_rate, optimizer
        )
        logger.info("Preparing SVI...")
        svi = self._setup_svi(optimizer, n_particles, scale=True)
        logger.info("Preparing training data...")
        (
            train_obs,
            mask_obs,
            train_covs,
            train_prior_scales,
        ) = self._setup_training_data()

        if self._informed:
            train_prior_scales = train_prior_scales.to(self.device)

        if batch_size < self.n_all_samples:
            logger.info("Using batches of size %s.", batch_size)
            tensors = (train_obs, mask_obs)
            if self.covariates is not None:
                tensors += (train_covs,)
            data_loader = DataLoader(
                TensorDataset(*tensors),
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=str(self.device) != "cpu",
                drop_last=False,
            )

            def _step():
                iteration_loss = 0
                for _, (sample_idx, tensors) in enumerate(data_loader):
                    iteration_loss += svi.step(
                        sample_idx.to(self.device),
                        *[tensor.to(self.device) for tensor in tensors],
                        prior_scales=train_prior_scales,
                    )
                return iteration_loss

        else:
            logger.info("Using complete dataset.")
            train_obs = train_obs.to(self.device)
            mask_obs = mask_obs.to(self.device)

            if train_covs is not None:
                train_covs = train_covs.to(self.device)

            def _step():
                return svi.step(
                    None, train_obs, mask_obs, train_covs, train_prior_scales
                )

        self.seed = seed
        if seed is not None:
            logger.info("Setting training seed to %s", seed)
            pyro.set_rng_seed(seed)
        # clean start
        logger.info("Cleaning parameter store")
        pyro.enable_validation(True)
        pyro.clear_param_store()

        logger.info("Starting training...")
        self._trained = True
        stop_early = False
        history = []
        pbar = range(n_epochs)
        if verbose > 0:
            pbar = tqdm(pbar)
            window_size = 5
        for epoch_idx in pbar:
            epoch_loss = _step()
            history.append(epoch_loss)
            if verbose > 0:
                if epoch_idx % window_size == 0 or epoch_idx == n_epochs - 1:
                    pbar.set_postfix({"ELBO": epoch_loss})
            if callbacks is not None:
                # TODO: dont really like this, a bit sloppy
                stop_early = any([callback(history) for callback in callbacks])
                if stop_early:
                    break

        # reset cache in case it was initialized by any of the callbacks
        self._cache = None
        return history, stop_early


class MuVIModel(PyroModule):
    def __init__(
        self,
        n_samples: List[int],
        n_subsamples: int,
        n_features: List[int],
        n_factors: int,
        n_covariates: List[int],
        likelihoods: List[str],
        global_prior_scale: float = 1.0,
        device: bool = None,
    ):
        """MuVI generative model.

        Parameters
        ----------
        n_samples : List[int]
            Number of samples as list for each group
        n_subsamples : int
            Number of subsamples (batch size)
        n_features : List[int]
            Number of features as list for each view
        n_factors : int
            Number of latent factors
        n_covariates : List[int]
            Number of covariates as list for each group
        likelihoods : List[str], optional
            List of likelihoods for each view,
            either "normal" or "bernoulli", by default None
        global_prior_scale (unused) : float, optional
            Determine the level of global sparsity, by default 1.0
        """
        super().__init__(name="MuVIModel")
        self.n_samples = n_samples
        self.n_subsamples = n_subsamples
        self.n_features = n_features

        self.sample_offsets = [0] + np.cumsum(self.n_samples).tolist()
        self.feature_offsets = [0] + np.cumsum(self.n_features).tolist()

        self.n_groups = len(self.n_samples)
        self.n_views = len(self.n_features)

        self.n_factors = n_factors
        self.n_covariates = n_covariates
        self.likelihoods = likelihoods
        self.same_likelihood = len(set(self.likelihoods)) == 1
        self.global_prior_scale = (
            1.0 if global_prior_scale is None else global_prior_scale
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
            "group": {"name": "group", "size": self.n_groups, "dim": -1},
            "view": {"name": "view", "size": self.n_views, "dim": -1},
            "factor": {"name": "factor", "size": self.n_factors, "dim": -2},
            "feature": {"name": "feature", "size": sum(self.n_features), "dim": -1},
            "sample": {"name": "sample", "size": sum(self.n_samples), "dim": -2},
            "covariate": {"name": "covariate", "size": self.n_covariates, "dim": -2},
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

        view_plate = self.get_plate("view")
        group_plate = self.get_plate("group")
        factor_plate = self.get_plate("factor")
        feature_plate = self.get_plate("feature")
        sample_plate = self.get_plate("sample", subsample=sample_idx)

        with group_plate:
            output_dict["group_scale"] = pyro.sample(
                "group_scale", dist.HalfCauchy(self._ones(1))
            )
            with factor_plate:
                output_dict["group_factor_scale"] = pyro.sample(
                    "group_factor_scale",
                    dist.HalfCauchy(self._ones(1)),
                )

        with view_plate:
            output_dict["view_scale"] = pyro.sample(
                "view_scale", dist.HalfCauchy(self._ones(1))
            )
            with factor_plate:
                output_dict["view_factor_scale"] = pyro.sample(
                    "view_factor_scale",
                    dist.HalfCauchy(self._ones(1)),
                )

        with feature_plate:
            with factor_plate:
                output_dict["local_scale"] = pyro.sample(
                    "local_scale",
                    dist.HalfCauchy(self._ones(1)),
                )
                output_dict["caux"] = pyro.sample(
                    "caux",
                    dist.InverseGamma(0.5 * self._ones(1), 0.5 * self._ones(1)),
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
                            * output_dict["view_factor_scale"][..., m : m + 1]
                            * output_dict["view_scale"][..., m : m + 1]
                        )
                        for m in range(self.n_views)
                    ],
                    -1,
                )

                output_dict["w"] = pyro.sample(
                    "w",
                    dist.Normal(
                        self._zeros(1),
                        (self.global_prior_scale * c * lmbda)
                        / torch.sqrt(c**2 + lmbda**2),
                    ),
                )

            if self.n_covariates > 0:
                with self.get_plate("covariate"):
                    output_dict["beta"] = pyro.sample(
                        "beta",
                        dist.Normal(self._zeros(1), self._ones(1)),
                    )

            output_dict["sigma"] = pyro.sample(
                "sigma",
                dist.InverseGamma(self._ones(1), self._ones(1)),
            )

        with sample_plate:
            lmbda_factor = torch.cat(
                [
                    (
                        self._ones((self.n_factors, sum(self.n_samples)))[
                            ...,
                            self.sample_offsets[g] : self.sample_offsets[g + 1],
                        ]
                        * output_dict["group_factor_scale"][..., g : g + 1]
                        * output_dict["group_scale"][..., g : g + 1]
                    )
                    for g in range(self.n_groups)
                ],
                -1,
            )
            if len(lmbda_factor.shape) > 2:
                lmbda_factor = lmbda_factor.view(
                    -1, sum(self.n_samples), self.n_factors
                )
            else:
                lmbda_factor = lmbda_factor.T
            output_dict["lmbda_factor"] = lmbda_factor
            output_dict["z"] = pyro.sample(
                "z",
                dist.Normal(self._zeros(self.n_factors), lmbda_factor),
            )

            y_loc = torch.matmul(output_dict["z"], output_dict["w"])
            if self.n_covariates > 0:
                y_loc = y_loc + torch.matmul(covs, output_dict["beta"])

            likelihoods = self.likelihoods
            feature_offsets = self.feature_offsets
            if self.same_likelihood:
                likelihoods = [likelihoods[0]]
                feature_offsets = [0, feature_offsets[-1]]

            ys = []
            for view_idx, likelihood in enumerate(likelihoods):
                feature_idx = slice(
                    feature_offsets[view_idx],
                    feature_offsets[view_idx + 1],
                )
                if likelihood == "normal":
                    y_dist = dist.Normal(
                        y_loc[..., feature_idx],
                        torch.sqrt(output_dict["sigma"][..., feature_idx]),
                    )
                else:
                    y_dist = dist.Bernoulli(logits=y_loc[..., feature_idx])

                with pyro.poutine.mask(mask=mask[..., feature_idx]):
                    ys.append(
                        pyro.sample(
                            f"y_{view_idx}",
                            y_dist,
                            obs=obs[..., feature_idx],
                            infer={"is_auxiliary": True},
                        )
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
        n_samples = sum(self.model.n_samples)
        n_features = sum(self.model.n_features)
        n_groups = self.model.n_groups
        n_views = self.model.n_views
        n_factors = self.model.n_factors

        site_to_shape = {
            "z": (n_samples, n_factors),
            "group_scale": n_groups,
            "group_factor_scale": (n_factors, n_groups),
            "view_scale": n_views,
            "view_factor_scale": (n_factors, n_views),
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

    def get_view_factor_scale(self):
        """Get the factor scales."""
        return self._get_map_estimate("view_factor_scale", False).T

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
        sample_idx: torch.Tensor,
        obs: torch.Tensor,
        mask: torch.Tensor,
        covs: torch.Tensor = None,
        prior_scales: torch.Tensor = None,
    ):
        """Approximate posterior."""
        output_dict = {}

        view_plate = self.model.get_plate("view")
        group_plate = self.model.get_plate("group")
        factor_plate = self.model.get_plate("factor")
        feature_plate = self.model.get_plate("feature")
        sample_plate = self.model.get_plate("sample", subsample=sample_idx)

        with group_plate:
            output_dict["group_scale"] = self._sample_log_normal("group_scale")
            with factor_plate:
                output_dict["group_factor_scale"] = self._sample_log_normal(
                    "group_factor_scale"
                )

        with view_plate:
            output_dict["view_scale"] = self._sample_log_normal("view_scale")
            with factor_plate:
                output_dict["view_factor_scale"] = self._sample_log_normal(
                    "view_factor_scale"
                )

        with feature_plate:
            with factor_plate:
                output_dict["local_scale"] = self._sample_log_normal("local_scale")
                output_dict["caux"] = self._sample_log_normal("caux")
                output_dict["w"] = self._sample_normal("w")

            if self.model.n_covariates > 0:
                with self.model.get_plate("covariate"):
                    output_dict["beta"] = self._sample_normal("beta")

            output_dict["sigma"] = self._sample_log_normal("sigma")

        with sample_plate as indices:
            z_loc, z_scale = self._get_loc_and_scale("z")
            if indices is not None:
                # indices = indices.to(self.device)
                z_loc = z_loc.index_select(0, indices)
                z_scale = z_scale.index_select(0, indices)
            output_dict["z"] = pyro.sample("z", dist.Normal(z_loc, z_scale))
            # output_dict["z"] = self._sample_normal("z")
        return output_dict


def save(model, dir_path="."):
    model_path = os.path.join(dir_path, "model.pkl")
    params_path = os.path.join(dir_path, "params.save")
    if os.path.isfile(model_path):
        logger.warning("`%s` already exists, overwriting.", model_path)
    if os.path.isfile(params_path):
        logger.warning("`%s` already exists, overwriting.", params_path)
    os.makedirs(dir_path, exist_ok=True)
    # if not os.path.isdir(os.path.dirname(dir_path)) and (
    #     os.path.dirname(dir_path) != ""
    # ):
    #     os.makedirs(dir_path)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    pyro.get_param_store().save(params_path)


def load(dir_path=".", with_params=True):
    model_path = os.path.join(dir_path, "model.pkl")
    params_path = os.path.join(dir_path, "params.save")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if with_params:
        pyro.get_param_store().load(params_path)
    # model = pyro.module("MuVI", model, update_module_params=True)
    return model
