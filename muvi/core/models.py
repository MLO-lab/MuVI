import logging
import time

from importlib.metadata import version
from typing import Any
from typing import Callable
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
from tabulate import tabulate
from tensordict import TensorDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from muvi.core.early_stopping import EarlyStoppingCallback
from muvi.core.index import _make_index_unique
from muvi.core.index import _normalize_index


logger = logging.getLogger(__name__)

Index = Union[int, str, list[int], list[str], np.ndarray, pd.Index]
SingleView = Union[np.ndarray, pd.DataFrame]
MultiView = Union[dict[str, SingleView], list[SingleView]]


def identity(x):
    return x


class MuVI(PyroModule):
    def __init__(
        self,
        observations: MultiView,
        prior_masks: Optional[MultiView] = None,
        covariates: Optional[SingleView] = None,
        prior_confidence: Optional[Union[float, str]] = "low",
        n_factors: Optional[int] = None,
        view_names: Optional[list[str]] = None,
        likelihoods: Optional[Union[dict[str, str], list[str]]] = None,
        reg_hs: bool = True,
        nmf: Optional[Union[dict[str, bool], list[bool]]] = None,
        pos_transform: str = "relu",
        normalize: bool = True,
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
        prior_confidence : float or string, optional
            Confidence of prior belief from 0 to 1 (exclusive),
            typical values are 'low' (0.99), 'med' (0.995) and 'high' (0.999),
            by default 'med'
        n_factors : int, optional
            Number of the uninformed latent factors,
            can be omitted when providing prior masks,
            or it can be used to introduce additional dense factors,
            by default None
        view_names : list[str], optional
            List of names for each view,
            determines view order as well,
            by default None
        likelihoods : Union[dict[str, str], list[str]], optional
            Likelihoods for each view,
            either "normal" or "bernoulli",
            by default None (all "normal")
        reg_hs : bool, optional
            Whether to use the regularized version of HS,
            by default True,
            only relevant when NOT using informative priors
        nmf : Union[dict[str, bool], list[bool]], optional
            Whether to use non-negative matrix factorization,
            by default False
        normalize : bool, optional
            Whether to normalize observations,
            by default True
        device : str, optional
            Device to run computations on, by default "cuda" (GPU)
        """
        super().__init__(name="MuVI")
        self.observations = self._setup_observations(observations, view_names)
        self.prior_confidence = self._setup_prior_confidence(prior_confidence)
        self.prior_masks, self.prior_scales = self._setup_prior_masks(
            prior_masks, n_factors
        )
        self.covariates = self._setup_covariates(covariates)
        self.likelihoods = self._setup_likelihoods(likelihoods)
        self.nmf = self._setup_nmf(nmf)
        self.pos_transform = pos_transform

        self.reg_hs = reg_hs
        self._informed = self.prior_masks is not None
        if not self.reg_hs and self._informed:
            logger.warning(
                "Informative priors require regularized horseshoe, "
                "setting `reg_hs` to True."
            )
            self.reg_hs = True

        self.normalize = normalize
        if self.normalize:
            self.observations = self._normalize_observations()

        self.device = self._setup_device(device)
        self.to(self.device)

        self._old_factor_names = self._factor_names.copy()

        self._model = None
        self._guide = None
        self._built = False
        self._trained = False
        self._training_log: dict[str, Any] = {}

        self._version = version("muvi")

        self._reset_cache()
        self._reset_factors()

    def __repr__(self):
        table = [
            ["n_views", self.n_views],
            ["n_samples", self.n_samples],
            [
                "n_features",
                ", ".join([f"{vn}: {self.n_features[vn]}" for vn in self.view_names]),
            ],
            ["n_factors", self.n_factors],
            ["prior_confidence", self.prior_confidence],
            ["n_covariates", self.n_covariates],
            [
                "likelihoods",
                ", ".join([f"{vn}: {self.likelihoods[vn]}" for vn in self.view_names]),
            ],
            ["nmf", ", ".join([f"{vn}: {self.nmf[vn]}" for vn in self.view_names])],
            ["reg_hs", self.reg_hs],
        ]

        if any(self.nmf.values()):
            table.append(["pos_transform", self.pos_transform])
        table.append(["device", self.device])

        header = f" MuVI version {self._version} "
        body = tabulate(table, headers=["Parameter", "Value"], tablefmt="github")
        row_len = len(body.split("\n")[0])
        left_padding_len = (row_len - len(header) - 2) // 2

        empty_line = "|" + "=" * (row_len - 2) + "|\n"
        header = "|" + " " * left_padding_len + header
        header = (
            empty_line + header + " " * (row_len - len(header) - 1) + "|\n" + empty_line
        )
        output = header + body + "\n" + empty_line
        return output

    @property
    def factor_order(self):
        return self._factor_order

    @factor_order.setter
    def factor_order(self, value):
        self._factor_order = self._factor_order[np.array(value)]
        if self._cache is not None:
            self._cache.reorder_factors(self._factor_order)

    @property
    def factor_names(self):
        return self._factor_names[self.factor_order]

    @factor_names.setter
    def factor_names(self, value):
        self._factor_names = _make_index_unique(pd.Index(value))
        if self._cache is not None:
            self._cache.rename_factors(self.factor_names)

    @property
    def factor_signs(self):
        return self._factor_signs[self.factor_order]

    @factor_signs.setter
    def factor_signs(self, value):
        self._factor_signs = value

    def _reset_cache(self):
        self._cache = None

    def _reset_factors(self):
        self._factor_order = np.arange(self.n_factors)
        self._factor_names = self._old_factor_names.copy()
        self._factor_signs = np.ones(self.n_factors, dtype=np.float32)

    def _compute_factor_signs(self):
        self.factor_signs = np.ones(self.n_factors, dtype=np.float32)

        # if nmf is enabled, all factors are positive
        if any(self.nmf.values()):
            return self.factor_signs
        # only compute signs if model is trained
        # and there is no nmf constraint
        if self._trained:
            w = self._guide.get_w()
            # TODO: looking at top loadings works better than including all
            # 100 feels a bit arbitrary though
            w = np.array(
                list(map(lambda x, y: y[x], np.argsort(-np.abs(w), axis=1)[:, :100], w))
            )
            self.factor_signs = (w.sum(axis=1) > 0) * 2 - 1

        return self.factor_signs

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
            idx = idx.astype(str)
        if not idx.is_unique:
            logger.warning("Making index unique.")
            idx = _make_index_unique(idx)
        return idx

    def _normalize_observations(self):
        logger.info("Normalizing observations.")
        for vn in self.observations:
            if self.likelihoods[vn] == "bernoulli":
                logger.info(
                    f"Skipping normalization for view `{vn}` with a Bernoulli"
                    " likelihood."
                )
                continue
            if self.nmf[vn]:
                logger.info(f"Setting min value of view `{vn}` to 0.")
                self.observations[vn] -= np.nanmin(self.observations[vn], axis=0)
            else:
                logger.info(f"Centering features of view `{vn}`.")
                self.observations[vn] -= np.nanmean(self.observations[vn], axis=0)
            global_std = np.nanstd(self.observations[vn])
            logger.info(
                f"Setting global standard deviation to 1.0 (from {global_std:.3f})."
            )
            self.observations[vn] /= global_std

        return self.observations

    def _merge(self, matrix_collection, method="union"):
        all_array = all(
            isinstance(matrix, np.ndarray) for matrix in matrix_collection.values()
        )
        all_dataframe = all(
            isinstance(matrix, pd.DataFrame) for matrix in matrix_collection.values()
        )

        if not all_array and not all_dataframe:
            raise ValueError(
                "All input must be of the same type, either np.ndarray or pd.DataFrame."
            )

        if all_array:
            matrix_collection = {
                k: pd.DataFrame(matrix) for k, matrix in matrix_collection.items()
            }

        key_list = []
        value_list = []
        for k, v in matrix_collection.items():
            key_list.append(k)
            value_list.append(v)

        index_offsets = [0, *list(np.cumsum([v.shape[1] for v in value_list]))]

        if method == "intersection":
            merged_matrix_collection = pd.concat(value_list, axis=1, join="inner")
        elif method == "union":
            merged_matrix_collection = pd.concat(value_list, axis=1, join="outer")
        else:
            raise ValueError(
                'Invalid merge method. Please choose either "intersection" or "union".'
            )

        merged_matrix_collection = {
            k: merged_matrix_collection.iloc[
                :, index_offsets[i] : index_offsets[i + 1]
            ].copy()
            for i, k in enumerate(key_list)
        }

        if all_array:
            merged_matrix_collection = {
                k: v.to_numpy(dtype=np.float32)
                for k, v in merged_matrix_collection.items()
            }

        return merged_matrix_collection

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

        observations = self._merge({vn: observations[vn] for vn in view_names})

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
        # from now on only working with dictonaries
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

    def _setup_prior_confidence(self, prior_confidence):
        low = 0.99
        med = 0.995
        high = 0.999
        if prior_confidence is None:
            logger.info(
                f"No prior confidence provided, setting `prior_confidence` to {med}."
            )
            prior_confidence = med

        if isinstance(prior_confidence, str):
            try:
                prior_confidence = {"low": low, "med": med, "high": high}[
                    prior_confidence.lower()
                ]
            except KeyError as e:
                raise KeyError(
                    "Invalid prior confidence, please provide one of `low`, `med`,"
                    " `high` or a positive value less than 1."
                ) from e

        if not (0 < prior_confidence < 1.0):
            raise ValueError(
                "Invalid prior confidence, please provide a positive value less than 1."
            )

        return prior_confidence

    def _setup_prior_masks(self, masks, n_factors):
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
            self.n_dense_factors = n_factors
            # TODO: duplicate line...see below
            self._factor_names = pd.Index([f"factor_{k}" for k in range(n_factors)])
            return None, None

        # if list convert to dict
        if isinstance(masks, list):
            masks = {self.view_names[m]: mask for m, mask in enumerate(masks)}

        masks = self._merge(masks)

        informed_views = []
        for vn in self.view_names:
            if vn in masks and np.any(masks[vn]):
                informed_views.append(vn)

        n_prior_factors = masks[informed_views[0]].shape[0]

        if n_factors is None:
            n_factors = 0

        n_dense_factors = n_factors
        n_factors += n_prior_factors

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
                    f"Mask of `{vn}` has {view_mask.shape[0]} factors "
                    f"instead of {n_prior_factors}, "
                    "all masks must have the same number of factors."
                )

            n_features_view = self.n_features[vn]

            if isinstance(view_mask, np.ndarray):
                logger.info("np.ndarray detected.")
                n_features_mask = view_mask.shape[1]
                if n_features_mask != n_features_view:
                    logger.warning(
                        f"Mask `{vn}` has {n_features_mask} features "
                        f"instead of {n_features_view}, "
                        "matching mask features to the view."
                    )

                    if n_features_mask > n_features_view:
                        # clip to the number of view features
                        logger.info(
                            "Mask has more features than expected, "
                            f"keeping only the first {n_features_view} features."
                        )
                        view_mask = np.copy(view_mask[:, :n_features_view])
                    else:
                        # pad with False to match the number of view features
                        logger.info(
                            "Mask has fewer features than expected, "
                            "padding the remaining features with False."
                        )
                        view_mask = np.concatenate(
                            [
                                view_mask,
                                np.zeros(
                                    (
                                        view_mask.shape[0],
                                        n_features_view - n_features_mask,
                                    )
                                ),
                            ],
                            axis=1,
                        )

            if isinstance(view_mask, pd.DataFrame):
                logger.info("pd.DataFrame detected.")

                feature_intersection = [
                    fn for fn in self.feature_names[vn] if fn in view_mask.columns
                ]
                n_feature_intersection = len(feature_intersection)
                if n_feature_intersection == 0:
                    raise ValueError(
                        f"None of the feature names for mask `{vn}` "
                        "match the feature names of its corresponding view. "
                        "Possible indication of passing np.array observations "
                        "with pd.DataFrame prior masks."
                    )
                if n_feature_intersection < n_features_view:
                    # pad with False to match the number of view features
                    logger.info(
                        "Mask has fewer features than expected, "
                        "padding the remaining features with False."
                    )
                    missing_features = [
                        fn
                        for fn in self.feature_names[vn]
                        if fn not in view_mask.columns
                    ]
                    view_mask = pd.concat(
                        [
                            view_mask,
                            pd.DataFrame(
                                False,
                                index=view_mask.index,
                                columns=missing_features,
                            ),
                        ],
                        axis=1,
                    )

                    feature_intersection += missing_features

                # otherwise simply subset features to the ones present
                view_mask = view_mask.loc[:, feature_intersection].copy()
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
                    view_mask = view_mask.loc[factor_names, :]

                if any(self.feature_names[vn] != view_mask.columns):
                    logger.info(
                        f"Feature names for mask `{vn}` "
                        "do not match the feature names of its corresponding view, "
                        "sorting names according to the view features."
                    )
                    view_mask = view_mask.loc[:, self.feature_names[vn]]

            masks[vn] = view_mask

        if factor_names is None:
            factor_names = [f"factor_{k}" for k in range(n_prior_factors)]
        if n_dense_factors > 0:
            factor_names = list(factor_names) + [
                f"dense_{k}" for k in range(n_dense_factors)
            ]
        self._factor_names = self._validate_index(pd.Index(factor_names))

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
                    [vm, np.ones((n_dense_factors, self.n_features[vn])).astype(bool)],
                    axis=0,
                )
                for vn, vm in masks.items()
            }

        prior_scales = {
            vn: np.clip(
                vm.astype(np.float32) + (1.0 - self.prior_confidence), 1e-8, 1.0
            )
            for vn, vm in prior_masks.items()
        }

        if n_dense_factors > 0:
            dense_scale = 1.0
            for vn in self.view_names:
                prior_scales[vn][n_prior_factors:, :] = dense_scale

        self.n_factors = n_factors
        self.n_dense_factors = n_dense_factors
        self.informed_views = informed_views
        return prior_masks, prior_scales

    def _setup_likelihoods(self, likelihoods):
        if likelihoods is None:
            likelihoods = ["normal" for _ in range(self.n_views)]
        if isinstance(likelihoods, list):
            likelihoods = {
                self.view_names[m]: ll.lower() for m, ll in enumerate(likelihoods)
            }
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

        if isinstance(covariates, np.ndarray):
            logger.info("np.ndarray detected.")
            n_samples_cov = covariates.shape[0]
            if n_samples_cov != self.n_samples:
                logger.warning(
                    f"Covariates have {n_samples_cov} samples "
                    f"instead of {self.n_samples}, matching samples."
                )

                if n_samples_cov > self.n_samples:
                    logger.info(
                        "Covariates have more samples than expected, "
                        f"keeping only the first {n_samples_cov} samples."
                    )
                    covariates = np.copy(covariates[: self.n_samples, :])
                else:
                    raise ValueError(
                        "Covariates have fewer samples than expected, "
                        "current version does not handle missing covariates"
                    )

        if isinstance(covariates, pd.DataFrame):
            logger.info("pd.DataFrame detected.")

            sample_intersection = [
                sn for sn in self.sample_names if sn in covariates.index
            ]
            n_sample_intersection = len(sample_intersection)
            if n_sample_intersection == 0:
                raise ValueError(
                    "None of the sample names for the covariates "
                    "match the sample names of the observations. "
                    "Possible indication of passing np.array observations "
                    "with pd.DataFrame covariates."
                )
            if n_sample_intersection < self.n_samples:
                raise ValueError(
                    "Covariates have fewer samples than expected, "
                    "current version does not handle missing covariates"
                )

            covariates = covariates.loc[sample_intersection, :].copy()

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
        self.covariate_names = self._validate_index(covariate_names)
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
        feature_idx: Union[Index, list[Index], dict[str, Index]] = "all",
        as_df: bool = False,
    ):
        """Get observations.

        Parameters
        ----------
        view_idx : Index, optional
            View index, by default "all"
        sample_idx : Index, optional
            Sample index, by default "all"
        feature_idx : Union[Index, list[Index], dict[str, Index]], optional
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
        feature_idx: Union[Index, list[Index], dict[str, Index]] = "all",
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
        feature_idx : Union[Index, list[Index], dict[str, Index]], optional
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
        feature_idx: Union[Index, list[Index], dict[str, Index]] = "all",
        as_df: bool = False,
    ):
        """Get imputed observations.

        Parameters
        ----------
        view_idx : Index, optional
            View index, by default "all"
        sample_idx : Index, optional
            Sample index, by default "all"
        feature_idx : Union[Index, list[Index], dict[str, Index]], optional
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
        feature_idx: Union[Index, list[Index], dict[str, Index]] = "all",
        as_df: bool = False,
    ):
        """Get prior masks.

        Parameters
        ----------
        view_idx : Index, optional
            View index, by default "all"
        factor_idx : Index, optional
            Factor index, by default "all"
        feature_idx : Union[Index, list[Index], dict[str, Index]], optional
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
            {k: v[self.factor_order, :] for k, v in self.prior_masks.items()},
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
        feature_idx: Union[Index, list[Index], dict[str, Index]] = "all",
        as_df: bool = False,
    ):
        """Get factor loadings.

        Parameters
        ----------
        view_idx : Index, optional
            View index, by default "all"
        factor_idx : Index, optional
            Factor index, by default "all"
        feature_idx : Union[Index, list[Index], dict[str, Index]], optional
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
        ws = [w[self.factor_order, :] * self.factor_signs[:, np.newaxis] for w in ws]
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
        feature_idx: Union[Index, list[Index], dict[str, Index]] = "all",
        as_df: bool = False,
    ):
        """Get factor loadings.

        Parameters
        ----------
        view_idx : Index, optional
            View index, by default "all"
        cov_idx : Index, optional
            Covariate index, by default "all"
        feature_idx : Union[Index, list[Index], dict[str, Index]], optional
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

        z = self._guide.get_z()
        z = z[:, self.factor_order] * self.factor_signs

        return self._get_sample_attr(
            z,
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

    def _setup_model_guide(self, scale_elbo: bool):
        """Setup model and guide.

        Parameters
        ----------
        scale_elbo : bool, optional
            Whether to scale the ELBO across views, by default True

        Returns
        -------
        bool
            Whether the build was successful
        """
        if not self._built:
            prior_scales = None
            if not self._informed:
                logger.warning(
                    "No prior feature sets provided, running model uninformed."
                )
            else:
                prior_scales = [
                    torch.Tensor(self.prior_scales[vn]) for vn in self.view_names
                ]

            self._model = MuVIModel(
                self.n_samples,
                n_features=[self.n_features[vn] for vn in self.view_names],
                n_factors=self.n_factors,
                prior_scales=prior_scales,
                n_covariates=self.n_covariates,
                likelihoods=[self.likelihoods[vn] for vn in self.view_names],
                reg_hs=self.reg_hs,
                nmf=[self.nmf[vn] for vn in self.view_names],
                pos_transform=self.pos_transform,
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

        # make sure keys match the arguments in the forward method of the MuVIModel
        training_data = {
            "obs": train_obs,
            "mask": mask_obs,
            "covs": train_covs,
        }
        return {k: v for k, v in training_data.items() if v is not None}

    def fit(
        self,
        batch_size: int = 0,
        n_epochs: int = 0,
        n_particles: int = 0,
        learning_rate: float = 0.005,
        optimizer: str = "clipped",
        scale_elbo: bool = True,
        early_stopping: bool = True,
        callbacks: Optional[list[Callable]] = None,
        verbose: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Perform inference.

        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default 0 (all samples)
        n_epochs : int, optional
            Number of iterations over the whole dataset,
            by default 0 (10k steps)
        n_particles : int, optional
            Number of particles/samples used to form the ELBO (gradient) estimators,
            by default 0 (1000 // batch_size)
        learning_rate : float, optional
            Learning rate, by default 0.005
        scale_elbo : bool, optional
            Whether to scale the ELBO across views, by default True
        optimizer : str, optional
            Optimizer as string, "adam" or "clipped", by default "clipped"
        early_stopping : bool, optional
            Whether to stop training early, by default True
        callbacks : list[Callable], optional
            List of callbacks during training, by default None
        verbose : bool, optional
            Whether to log progress, by default True
        seed : int, optional
            Training seed, by default None
        """

        # if invalid or out of bounds set to n_samples
        if batch_size is None or not (0 < batch_size <= self.n_samples):
            batch_size = self.n_samples

        n_batches_per_epoch = max(1, self.n_samples // batch_size)
        if n_epochs is None or n_epochs == 0:
            n_epochs = 10000 // n_batches_per_epoch

        if n_particles < 1:
            n_particles = max(1, 1000 // batch_size)
        if n_particles > 1:
            logger.info(f"Using {n_particles} particles in parallel.")
        logger.info("Preparing model and guide...")
        self._setup_model_guide(scale_elbo)
        logger.info("Preparing optimizer...")
        opt = self._setup_optimizer(batch_size, n_epochs, learning_rate, optimizer)
        logger.info("Preparing SVI...")
        svi = self._setup_svi(opt, n_particles, scale=True)
        logger.info("Preparing training data...")
        training_data = self._setup_training_data()

        min_epochs = kwargs.pop("min_epochs", n_epochs // 10)
        patience = kwargs.pop("patience", max(10, int(5 * n_batches_per_epoch)))
        early_stopping_callback = EarlyStoppingCallback(
            min_epochs, patience=patience, **kwargs
        )

        if callbacks is None:
            callbacks = []

        if batch_size < self.n_samples:
            logger.info(f"Using batches of size `{batch_size}`.")

            training_data = TensorDict(
                dict(training_data.items()), batch_size=[self.n_samples]
            )
            training_data["sample_idx"] = torch.arange(self.n_samples)

            data_loader = DataLoader(
                training_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                collate_fn=identity,
                pin_memory=str(self.device) != "cpu",
                drop_last=False,
            )

            def _step():
                iteration_loss = 0
                for _, tensor_dict in enumerate(data_loader):
                    iteration_loss += svi.step(**tensor_dict.to(self.device))
                return iteration_loss

        else:
            logger.info("Using complete dataset.")
            # move all data to device once
            training_data = {k: v.to(self.device) for k, v in training_data.items()}

            def _step():
                return svi.step(None, **training_data)

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
        # needs to be set here otherwise the logcallback fails
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
                for callback in callbacks:
                    callback(history)
                if early_stopping and early_stopping_callback(history):
                    stop_early = True
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
        self._post_fit()

    def _post_fit(self):
        """Post fit method."""
        self._trained = True
        self._reset_cache()
        self._reset_factors()
        self._compute_factor_signs()


class MuVIModel(PyroModule):
    def __init__(
        self,
        n_samples: int,
        n_features: list[int],
        n_factors: int,
        prior_scales: Optional[list[torch.Tensor]],
        n_covariates: int,
        likelihoods: list[str],
        global_prior_scale: float = 1.0,
        reg_hs: bool = True,
        nmf: Optional[list[bool]] = None,
        pos_transform=None,
        scale_elbo: bool = True,
        device=None,
    ):
        """MuVI generative model.

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_features : list[int]
            Number of features as list for each view
        n_factors : int
            Number of latent factors
        prior_scales : list[torch.Tensor], optional
            Local prior scales with prior information,
            by default None (uninformed)
        n_covariates : int
            Number of covariates
        likelihoods : list[str], optional
            List of likelihoods for each view,
            either "normal" or "bernoulli", by default None
        global_prior_scale : float, optional
            Determine the level of global sparsity, by default 1.0
        reg_hs : bool, optional
            Whether to use the regularized version of HS,
            by default True
        nmf : list[bool], optional
            Whether to use non-negative matrix factorization,
            by default empty (False for all views)
        scale_elbo : bool
            Whether to scale the ELBO across views, by default True
        device : str, optional
            Device to run computations on, by default "cuda" (GPU)
        """
        super().__init__(name="MuVIModel")
        self.n_samples = n_samples
        self.n_features = n_features
        self.feature_offsets = [0, *np.cumsum(self.n_features).tolist()]
        self.n_views = len(self.n_features)
        self.n_factors = n_factors
        self.prior_scales = prior_scales
        if self.prior_scales is not None:
            self.prior_scales = torch.cat(self.prior_scales, 1).to(device)
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
        self.pos_transform = None
        if pos_transform == "softplus":
            self.pos_transform = torch.nn.Softplus()
        if pos_transform == "relu":
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
            "factor_left": {
                "name": "factor_left",
                "size": self.n_factors,
                "dim": -2,
            },
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
                        if self.prior_scales is not None:
                            c = (
                                c
                                * self.prior_scales[
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
                        if self.nmf[m]:
                            output_dict[f"beta_{m}"] = self.pos_transform(
                                output_dict[f"beta_{m}"]
                            )

                output_dict[f"sigma_{m}"] = pyro.sample(
                    f"sigma_{m}",
                    dist.LogNormal(self._zeros((1,)), self._ones((1,))),
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
                            output_dict[f"sigma_{m}"],
                        )
                    else:
                        y_dist = dist.Bernoulli(logits=y_loc)

                    feature_idx = slice(
                        self.feature_offsets[m],
                        self.feature_offsets[m + 1],
                    )

                    with (
                        pyro.poutine.mask(mask=mask[..., feature_idx]),
                        pyro.poutine.scale(scale=self.view_scales[m]),
                    ):
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
