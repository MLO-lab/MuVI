import numpy as np
import pandas as pd
import pytest
import anndata as ad
import mudata as mu
from pandas.api.types import is_string_dtype

import muvi
from muvi import MuVI


def test_from_adata(pandas_input):
    view_idx = 0
    view_name = pandas_input["view_names"][view_idx]
    obs = pandas_input["observations"][view_idx]
    mask = pandas_input["masks"][view_idx]
    cov = pandas_input["covariates"]

    n_factors = mask.shape[0]

    adata = ad.AnnData(obs)
    adata.obsm["X_np"] = obs.iloc[:, : (obs.shape[1] // 2)].values
    adata.obsm["X_pd"] = obs.iloc[:, : (obs.shape[1] // 2)]
    adata.varm["prior_mask_np"] = mask.T.values
    adata.varm["prior_mask_pd"] = mask.T
    adata.obsm["covariate_np"] = cov.values
    adata.obsm["covariate_pd"] = cov

    models = {}
    for obs_key in ["", "X_np", "X_pd"]:
        models[obs_key] = {}
        for prior_mask_key in ["", "prior_mask_np", "prior_mask_pd"]:
            models[obs_key][prior_mask_key] = {}
            for covariate_key in ["", "covariate_np", "covariate_pd"]:
                if "np" in obs_key and (
                    "pd" in prior_mask_key or "pd" in covariate_key
                ):
                    continue

                obs_key_val = obs_key
                if obs_key_val == "":
                    obs_key_val = None

                prior_mask_key_val = prior_mask_key
                if prior_mask_key_val == "":
                    prior_mask_key_val = None

                covariate_key_val = covariate_key
                if covariate_key_val == "":
                    covariate_key_val = None

                model = muvi.tl.from_adata(
                    adata,
                    obs_key=obs_key_val,
                    prior_mask_key=prior_mask_key_val,
                    covariate_key=covariate_key_val,
                    n_factors=n_factors,
                )

                models[obs_key][prior_mask_key][covariate_key] = model
                assert model.n_views == 1
                assert model.n_samples == obs.shape[0]
                assert model.n_factors == mask.shape[0]
                if prior_mask_key != "":
                    assert (
                        model.n_features[view_name]
                        == model.prior_masks[view_name].shape[1]
                    )


def test_from_mdata(pandas_input):
    n_factors = pandas_input["masks"][0].shape[0]

    adata_dict = {}

    for view_idx, view_obs in enumerate(pandas_input["observations"]):
        adata = ad.AnnData(view_obs)
        adata.obsm["X_np"] = view_obs.iloc[:, : (view_obs.shape[1] // 2)].values
        adata.obsm["X_pd"] = view_obs.iloc[:, : (view_obs.shape[1] // 2)]
        adata.varm["prior_mask_np"] = pandas_input["masks"][view_idx].T.values
        adata.varm["prior_mask_pd"] = pandas_input["masks"][view_idx].T
        adata_dict[pandas_input["view_names"][view_idx]] = adata

    mdata = mu.MuData(adata_dict)

    mdata.obsm["covariate_np"] = pandas_input["covariates"].values
    mdata.obsm["covariate_pd"] = pandas_input["covariates"]

    models = {}
    for obs_key in ["", "X_np", "X_pd"]:
        models[obs_key] = {}
        for prior_mask_key in ["", "prior_mask_np", "prior_mask_pd"]:
            models[obs_key][prior_mask_key] = {}
            for covariate_key in ["", "covariate_np", "covariate_pd"]:
                if "np" in obs_key and (
                    "pd" in prior_mask_key or "pd" in covariate_key
                ):
                    continue

                obs_key_val = obs_key
                if obs_key_val == "":
                    obs_key_val = None

                prior_mask_key_val = prior_mask_key
                if prior_mask_key_val == "":
                    prior_mask_key_val = None

                covariate_key_val = covariate_key
                if covariate_key_val == "":
                    covariate_key_val = None

                model = muvi.tl.from_mdata(
                    mdata,
                    obs_key=obs_key_val,
                    prior_mask_key=prior_mask_key_val,
                    covariate_key=covariate_key_val,
                    n_factors=n_factors,
                )

                models[obs_key][prior_mask_key][covariate_key] = model

                assert (model.view_names == pandas_input["view_names"]).all()
                for view_idx, view_name in enumerate(model.view_names):
                    assert (
                        model.n_samples
                        == pandas_input["observations"][view_idx].shape[0]
                    )
                    assert model.n_factors == pandas_input["masks"][view_idx].shape[0]
                    if prior_mask_key != "":
                        assert (
                            model.n_features[view_name]
                            == model.prior_masks[view_name].shape[1]
                        )
