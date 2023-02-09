"""Data module."""
import itertools
import logging
import math
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DataGenerator:
    def __init__(
        self,
        n_samples: List[int] = None,
        n_features: List[int] = None,
        likelihoods: List[str] = None,
        group_factor_config: List[int] = None,
        view_factor_config: List[int] = None,
        n_covariates: int = 0,
        factor_size_params: Tuple[float] = None,
        factor_size_dist: str = "uniform",
        n_active_factors: float = 1.0,
        **kwargs,
    ) -> None:
        """Generate synthetic data

        Parameters
        ----------
        n_samples : List[int], optional
            Number of samples for each group, by default None
        n_features : List[int], optional
            Number of features for each view, by default None
        likelihoods : List[str], optional
            Likelihoods for each view, 'normal' or 'bernoulli', by default None
        group_factor_config : List[int], optional
            Factor sparsity config across groups as List of int
            [fully shared, partially shared, private factors],
            by default None
        view_factor_config : List[int], optional
            Factor sparsity config across views as List of int
            [fully shared, partially shared, private factors],
            the sum must match that of `group_factor_config`,
            by default None
        n_covariates : int, optional
            Number of observed covariates, by default 0
        factor_size_params : Tuple[float], optional
            Parameters for the distribution of the number
            of active factor loadings for the latent factors,
            by default None
        factor_size_dist : str, optional
            Distribution of the number of active factor loadings,
            either "uniform" or "gamma",
            by default "uniform"
        n_active_factors : float, optional
            Number or fraction of active factors, by default 1.0 (all)
        """

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_groups = len(self.n_samples)
        self.n_views = len(self.n_features)

        if group_factor_config is None:
            group_factor_config = [2, 15, 3]
        if view_factor_config is None:
            view_factor_config = [2, 15, 3]

        self.n_factors = sum(group_factor_config)
        if self.n_factors != sum(view_factor_config):
            raise ValueError("Group factor config must match view factor config.")

        self.n_group_fully_shared_factors = group_factor_config[0]
        self.n_group_partially_shared_factors = group_factor_config[1]
        self.n_group_private_factors = group_factor_config[2]

        self.n_view_fully_shared_factors = view_factor_config[0]
        self.n_view_partially_shared_factors = view_factor_config[1]
        self.n_view_private_factors = view_factor_config[2]

        self.n_covariates = n_covariates

        if factor_size_params is None:
            if factor_size_dist == "uniform":
                logger.warning(
                    "Using a uniform distribution with parameters 0.05 and 0.15 "
                    "for generating the number of active factor loadings."
                )
                factor_size_params = (0.05, 0.15)
            elif factor_size_dist == "gamma":
                logger.warning(
                    "Using a uniform distribution with shape of 1 and scale of 50 "
                    "for generating the number of active factor loadings."
                )
                factor_size_params = (1.0, 50.0)

        if isinstance(factor_size_params, tuple):
            factor_size_params = [factor_size_params for _ in range(self.n_views)]

        self.factor_size_params = factor_size_params
        self.factor_size_dist = factor_size_dist

        # custom assignment
        if likelihoods is None:
            likelihoods = ["normal" for _ in range(self.n_views)]
        self.likelihoods = likelihoods

        self.n_active_factors = n_active_factors

        # set upon data generation
        # covariates
        self.xs = None
        # covariate coefficients
        self.betas = None
        # latent factors
        self.zs = None
        # factor loadings
        self.ws = None
        self.sigmas = None
        self.ys = None
        self.w_masks = None
        self.noisy_w_masks = None
        self.active_factor_indices = None
        self.view_factor_mask = None
        # set when introducing missingness
        self.presence_masks = None

    def _to_matrix(self, matrix_list, axis=1):
        if any([isinstance(x, list) for x in matrix_list]):
            return np.concatenate([np.concatenate(y, axis=1) for y in matrix_list])
        return np.concatenate(matrix_list, axis=axis)

    def _attr_to_matrix(self, attr_name, axis):
        attr = getattr(self, attr_name)
        if attr is not None:
            attr = self._to_matrix(attr, axis=axis)
        return attr

    def _mask_to_nan(self):
        nan_masks = []
        for group_mask in self.presence_masks:
            nan_group_masks = []
            for mask in group_mask:
                nan_mask = np.array(mask, dtype=np.float32, copy=True)
                nan_mask[nan_mask == 0] = np.nan
                nan_group_masks.append(nan_mask)
            nan_masks.append(nan_group_masks)
        return nan_masks

    def _mask_to_bool(self):
        bool_masks = []
        for group_mask in self.presence_masks:
            bool_group_masks = []
            for mask in group_mask:
                bool_mask = mask == 1.0
                bool_group_masks.append(bool_mask)
        return bool_masks

    @property
    def missing_ys(self):
        if self.ys is None:
            logger.warning("Generate data first by calling `generate`.")
            return []
        if self.presence_masks is None:
            logger.warning(
                "Introduce missing data first by calling `generate_missingness`."
            )
            return self.ys

        nan_masks = self._mask_to_nan()

        return [
            [self.ys[g][m] * nan_masks[g][m] for m in range(self.n_views)]
            for g in range(self.n_groups)
        ]

    @property
    def y(self):
        return self._attr_to_matrix("ys", axis=1)

    @property
    def missing_y(self):
        return self._attr_to_matrix("missing_ys", axis=1)

    @property
    def z(self):
        return self._attr_to_matrix("zs", axis=0)
    
    @property
    def x(self):
        return self._attr_to_matrix("xs", axis=0)

    @property
    def w(self):
        return self._attr_to_matrix("ws", axis=1)

    @property
    def w_mask(self):
        return self._attr_to_matrix("w_masks", axis=1)

    @property
    def noisy_w_mask(self):
        return self._attr_to_matrix("noisy_w_masks", axis=1)

    def _generate_factor_mask(
        self,
        n_dim,
        n_fully_shared,
        n_partially_shared,
        n_private,
        rng=None,
        n_comb=None,
    ):
        # DEPRECATED
        # if n_comb is not None:
        #     logger.warning(
        #         "n_comb is not None, "
        #         "generating all possible binary combinations of %s variables",
        #         n_comb,
        #     )
        #     self.n_fully_shared_factors = 1
        #     self.n_private_factors = self.n_views
        #     self.n_partially_shared_factors = 2**n_comb - 2 - self.n_private_factors

        #     return np.array(
        #         [list(i) for i in itertools.product([1, 0], repeat=n_comb)]
        #     )[:-1, :].T
        if rng is None:
            rng = np.random.default_rng()

        factor_mask = np.ones([n_dim, self.n_factors])

        for factor_idx in range(n_fully_shared, self.n_factors):
            # exclude view subsets for partially shared factors
            if factor_idx < n_fully_shared + n_partially_shared:
                if n_dim > 2:
                    exclude_subset_size = rng.integers(1, n_dim - 1)
                else:
                    exclude_subset_size = 0

                exclude_subset = rng.choice(n_dim, exclude_subset_size, replace=False)
            # exclude all but one view for private factors
            else:
                include_idx = rng.integers(n_dim)
                exclude_subset = [i for i in range(n_dim) if i != include_idx]

            for j in exclude_subset:
                factor_mask[j, factor_idx] = 0

        if n_private >= n_dim:
            factor_mask[-n_dim:, -n_dim:] = np.eye(n_dim)

        return factor_mask

    def normalise(self, with_std=False):

        for g in range(self.n_groups):
            for m in range(self.n_views):
                if self.likelihoods[m] == "normal":
                    y = np.array(self.ys[g][m], dtype=np.float32, copy=True)
                    y -= y.mean(axis=0)
                    if with_std:
                        y_std = y.std(axis=0)
                        y = np.divide(y, y_std, out=np.zeros_like(y), where=y_std != 0)
                    self.ys[g][m] = y

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def generate(
        self, seed: int = None, n_comb: int = None, overwrite: bool = False
    ) -> None:

        rng = np.random.default_rng()

        if seed is not None:
            rng = np.random.default_rng(seed)

        if self.ys is not None and not overwrite:
            logger.warning(
                "Data has already been generated, "
                "to generate new data please set `overwrite` to True."
            )
            return rng

        group_factor_mask = self._generate_factor_mask(
            self.n_groups,
            self.n_group_fully_shared_factors,
            self.n_group_partially_shared_factors,
            self.n_group_private_factors,
            rng=rng,
            n_comb=n_comb,
        )

        view_factor_mask = self._generate_factor_mask(
            self.n_views,
            self.n_view_fully_shared_factors,
            self.n_view_partially_shared_factors,
            self.n_view_private_factors,
            rng=rng,
            n_comb=n_comb,
        )

        n_active_factors = self.n_active_factors
        if n_active_factors <= 1.0:
            # if fraction of active factors convert to int
            n_active_factors = int(n_active_factors * self.n_factors)

        active_factor_indices = sorted(
            rng.choice(
                self.n_factors,
                size=math.ceil(n_active_factors),
                replace=False,
            )
        )

        for factor_idx in range(self.n_factors):
            if factor_idx not in active_factor_indices:
                view_factor_mask[:, factor_idx] = 0.0

        xs = []
        zs = []
        betas = []
        ws = []
        w_masks = []
        sigmas = []

        tiny_threshold = 0.1

        for g in range(self.n_groups):
            n_samples = self.n_samples[g]
            x_shape = (n_samples, self.n_covariates)
            z_shape = (n_samples, self.n_factors)
            z = rng.standard_normal(z_shape) * group_factor_mask[g, :]
            if self.n_covariates > 0:
                x = rng.standard_normal(x_shape)

            # add some noise to avoid exactly zero values
            z = np.where(
                np.abs(z) < tiny_threshold,
                tiny_threshold + rng.standard_normal(z_shape) / 100,
                z,
            )

            zs.append(z)
            if self.n_covariates > 0:
                xs.append(x)
                
        for m in range(self.n_views):
            n_features = self.n_features[m]
            w_shape = (self.n_factors, n_features)
            w = rng.standard_normal(w_shape)
            w_mask = np.zeros(w_shape)

            fraction_active_features = {
                "gamma": lambda shape, scale: (
                    rng.gamma(shape, scale, self.n_factors) + 20
                )
                / n_features,
                "uniform": lambda low, high: rng.uniform(low, high, self.n_factors),
            }[self.factor_size_dist](
                self.factor_size_params[m][0], self.factor_size_params[m][1]
            )

            for factor_idx, faft in enumerate(fraction_active_features):
                if view_factor_mask[m, factor_idx] > 0:
                    w_mask[factor_idx] = rng.choice(2, n_features, p=[1 - faft, faft])

            # set small values to zero
            w_mask[np.abs(w) < tiny_threshold] = 0.0
            w = w_mask * w
            # add some noise to avoid exactly zero values
            w = np.where(
                np.abs(w) < tiny_threshold, w + rng.standard_normal(w_shape) / 100, w
            )
            assert ((np.abs(w) > tiny_threshold) * 1.0 == w_mask).all()

            if self.n_covariates > 0:
                beta_shape = (self.n_covariates, n_features)
                # reduce effect of betas by scaling them down
                beta = rng.standard_normal(beta_shape) / 10
                betas.append(beta)

            # generate feature sigmas
            sigma = 1.0 / np.sqrt(rng.gamma(10.0, 1.0, n_features))

            ws.append(w)
            sigmas.append(sigma)
            w_masks.append(w_mask)

        ys = []
        for g in range(self.n_groups):
            group_ys = []
            for m in range(self.n_views):
                y_loc = np.matmul(zs[g], ws[m])
                if self.n_covariates > 0:
                    y_loc = y_loc + np.matmul(xs[g], betas[m])
                if self.likelihoods[m] == "normal":
                    y = rng.normal(loc=y_loc, scale=sigmas[m])
                else:
                    y = rng.binomial(1, self.sigmoid(y_loc))
                group_ys.append(y)
            ys.append(group_ys)

        if self.n_covariates > 0:
            self.xs = xs
            self.betas = betas

        self.zs = zs
        self.ws = ws
        self.w_masks = w_masks
        self.sigmas = sigmas
        self.ys = ys
        self.active_factor_indices = active_factor_indices
        self.group_factor_mask = group_factor_mask
        self.view_factor_mask = view_factor_mask

        return rng

    def get_noisy_mask(self, rng=None, noise_fraction=0.1, informed_view_indices=None):
        if rng is None:
            rng = np.random.default_rng()

        if informed_view_indices is None:
            logger.warning(
                "Parameter `informed_view_indices` set to None, "
                "adding noise to all views."
            )
            informed_view_indices = list(range(self.n_views))

        noisy_w_masks = [np.array(mask, copy=True) for mask in self.w_masks]

        if len(informed_view_indices) == 0:
            logger.warning(
                "Parameter `informed_view_indices` "
                "set to an empty list, removing information from all views."
            )
            self.noisy_w_masks = [np.ones_like(mask) for mask in noisy_w_masks]
            return self.noisy_w_masks

        for m in range(self.n_views):
            noisy_w_mask = noisy_w_masks[m]

            if m in informed_view_indices:
                fraction_active_cells = (
                    noisy_w_mask.mean(axis=1).sum() / self.view_factor_mask[0].sum()
                )
                for factor_idx in range(self.n_factors):

                    active_cell_indices = noisy_w_mask[factor_idx, :].nonzero()[0]
                    # if all features turned off
                    # => simulate random noise in terms of false positives only
                    if len(active_cell_indices) == 0:
                        logger.warning(
                            "Factor %s is completely off, "
                            "inserting %.2f%% false positives.",
                            factor_idx,
                            (100 * fraction_active_cells),
                        )
                        active_cell_indices = rng.choice(
                            self.n_features[m],
                            int(self.n_features[m] * fraction_active_cells),
                            replace=False,
                        )

                    inactive_cell_indices = (
                        noisy_w_mask[factor_idx, :] == 0
                    ).nonzero()[0]
                    n_noisy_cells = int(noise_fraction * len(active_cell_indices))
                    swapped_indices = zip(
                        rng.choice(
                            len(active_cell_indices), n_noisy_cells, replace=False
                        ),
                        rng.choice(
                            len(inactive_cell_indices), n_noisy_cells, replace=False
                        ),
                    )

                    for on_idx, off_idx in swapped_indices:
                        noisy_w_mask[factor_idx, active_cell_indices[on_idx]] = 0.0
                        noisy_w_mask[factor_idx, inactive_cell_indices[off_idx]] = 1.0

            else:

                noisy_w_mask.fill(0.0)

        self.noisy_w_masks = noisy_w_masks
        return self.noisy_w_masks

    def generate_missingness(
        self,
        random_fraction: float = 0.0,
        n_partial_samples: int = 0,
        n_partial_features: int = 0,
        missing_fraction_partial_features: float = 0.0,
        seed=None,
    ):

        rng = np.random.default_rng()

        if seed is not None:
            rng = np.random.default_rng(seed)

        n_partial_samples = int(n_partial_samples)
        n_partial_features = int(n_partial_features)

        masks = []
        for g in range(self.n_groups):

            sample_view_mask = np.ones((self.n_samples[g], self.n_views))
            missing_sample_indices = rng.choice(
                self.n_samples[g], n_partial_samples, replace=False
            )

            # partially missing samples
            for ms_idx in missing_sample_indices:
                if self.n_views > 1:
                    exclude_view_subset_size = rng.integers(1, self.n_views)
                else:
                    exclude_view_subset_size = 0
                exclude_view_subset = rng.choice(
                    self.n_views, exclude_view_subset_size, replace=False
                )
                sample_view_mask[ms_idx, exclude_view_subset] = 0

            mask = np.repeat(sample_view_mask, self.n_features, axis=1)

            # partially missing features
            missing_feature_indices = rng.choice(
                sum(self.n_features), n_partial_features, replace=False
            )

            for mf_idx in missing_feature_indices:
                random_sample_indices = rng.choice(
                    self.n_samples[g],
                    int(self.n_samples[g] * missing_fraction_partial_features),
                    replace=False,
                )
                mask[random_sample_indices, mf_idx] = 0

            # remove random fraction
            mask *= rng.choice(
                [0, 1], mask.shape, p=[random_fraction, 1 - random_fraction]
            )

            view_feature_offsets = [0] + np.cumsum(self.n_features).tolist()
            group_masks = []
            for offset_idx in range(len(view_feature_offsets) - 1):
                start_offset = view_feature_offsets[offset_idx]
                end_offset = view_feature_offsets[offset_idx + 1]
                group_masks.append(mask[:, start_offset:end_offset])
            masks.append(group_masks)

        self.presence_masks = masks

        return rng

    def _permute_features(self, lst, new_order):
        return [np.array(lst[m][:, o], copy=True) for m, o in enumerate(new_order)]

    def _permute_factors(self, lst, new_order):
        return [np.array(lst[m][o, :], copy=True) for m, o in enumerate(new_order)]

    def permute_features(self, new_feature_order):
        if len(new_feature_order) != self.n_features:
            raise ValueError(
                "Length of new order list must equal the number of features."
            )

        if self.betas is not None:
            self.betas = self._permute_features(self.betas, new_feature_order)
        self.ws = self._permute_features(self.ws, new_feature_order)
        self.w_masks = self._permute_features(self.w_masks, new_feature_order)
        if self.noisy_w_masks is not None:
            self.noisy_w_masks = self._permute_features(
                self.noisy_w_masks, new_feature_order
            )
        self.sigmas = self._permute_features(self.sigmas, new_feature_order)
        self.ys = self._permute_features(self.ys, new_feature_order)
        if self.presence_masks is not None:
            self.missing_ys = self._permute_features(self.missing_ys, new_feature_order)
            self.presence_masks = self._permute_features(
                self.presence_masks, new_feature_order
            )

    def permute_factors(self, new_factor_order):
        if len(new_factor_order) != self.n_factors:
            raise ValueError(
                "Length of new order list must equal the number of factors."
            )

        self.z = np.array(self.z[:, np.array(new_factor_order)], copy=True)
        self.ws = self._permute_factors(self.ws, new_factor_order)
        self.w_masks = self._permute_factors(self.w_masks, new_factor_order)
        if self.noisy_w_masks is not None:
            self.noisy_w_masks = self._permute_factors(
                self.noisy_w_masks, new_factor_order
            )
        self.view_factor_mask = [
            self.view_factor_mask[m, np.array(new_factor_order)]
            for m in range(self.n_views)
        ]
        self.active_factor_indices = np.array(
            self.active_factor_indices[np.array(new_factor_order)], copy=True
        )
