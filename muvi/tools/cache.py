import logging

import anndata as ad
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class Cache:
    FILTERED_KEY = "filtered_scores"
    META_KEY = "metadata"

    METRIC_RMSE = "rmse"
    METRIC_R2 = "r2"

    TEST_POS = "pos"
    TEST_NEG = "neg"
    TEST_ALL = "all"
    TEST_P = "p"
    TEST_P_ADJ = "p_adj"
    TEST_T = "t"

    UNS_GROUPED_R2 = "grouped_r2"

    def __init__(self, model) -> None:
        self.clear()
        self.setup(model)

    def clear(self):
        self.uns = {}
        self.factor_adata = None
        self.cov_adata = None
        self.use_rep = "X"

    @property
    def factor_metadata(self):
        if self.factor_adata is None:
            return None
        return self.factor_adata.varm[Cache.META_KEY]

    @property
    def cov_metadata(self):
        if self.cov_adata is None:
            return None
        return self.cov_adata.varm[Cache.META_KEY]

    def setup(self, model):
        self._setup_uns()
        self._setup_adata(model)

    def _setup_uns(self):
        for view_score_key in [Cache.METRIC_R2, Cache.METRIC_RMSE]:
            if view_score_key not in self.uns:
                self.uns[f"view_{view_score_key}"] = {}

    def _setup_adata(self, model):
        columns = []
        for key in [Cache.METRIC_R2, Cache.METRIC_RMSE]:
            for vn in model.view_names:
                columns.append(f"{key}_{vn}")

        for key in [Cache.TEST_P, Cache.TEST_P_ADJ, Cache.TEST_T]:
            for sign in [Cache.TEST_ALL, Cache.TEST_NEG, Cache.TEST_POS]:
                for vn in model.view_names:
                    columns.append(f"{key}_{sign}_{vn}")

        if model.n_factors > 0:
            self.factor_adata = ad.AnnData(
                model.get_factor_scores(as_df=True), dtype=np.float32
            )
            self.factor_adata.varm[Cache.META_KEY] = pd.DataFrame(
                index=self.factor_adata.var_names, columns=columns, dtype=np.float32
            )

        if model.n_covariates > 0:
            self.cov_adata = ad.AnnData(
                model.get_covariates(as_df=True), dtype=np.float32
            )
            self.cov_adata.varm[Cache.META_KEY] = pd.DataFrame(
                index=self.cov_adata.var_names, columns=columns, dtype=np.float32
            )

    def update_uns(self, key, scores):
        self.uns[key].update(scores)

    def update_factor_metadata(self, scores):
        if self.factor_adata is not None:
            self.factor_adata.varm[Cache.META_KEY].update(scores.astype(np.float32))

    def update_cov_metadata(self, scores):
        if self.cov_adata is not None:
            self.cov_adata.varm[Cache.META_KEY].update(scores.astype(np.float32))

    def reorder_factors(self, order):
        if self.factor_adata is not None:
            self.factor_adata = self.factor_adata[:, order].copy()

    def rename_factors(self, factor_names):
        if self.factor_adata is not None:
            self.factor_adata.var_names = factor_names

    def filter_factors(self, factor_idx):
        self.factor_adata.obsm[Cache.FILTERED_KEY] = (
            self.factor_adata.to_df().loc[:, factor_idx].copy()
        )
        self.use_rep = Cache.FILTERED_KEY
        uns_keys = list(self.factor_adata.uns.keys())

        # remove neighborhood information
        if "neighbors" in uns_keys:
            logger.warning("Removing old neighborhood graph.")
            self.factor_adata.uns.pop("neighbors", None)
            logger.warning("Removing old distances.")
            self.factor_adata.obsp.pop("distances", None)
            logger.warning("Removing old connectivities.")
            self.factor_adata.obsp.pop("connectivities", None)

        # remove dendrogram information
        for key in uns_keys:
            if "dendrogram" in key:
                logger.warning("Removing old dendrogram.")
                self.factor_adata.uns.pop(key, None)

        logger.info("Factors filtered successfully.")

        return True
