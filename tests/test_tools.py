import muvi
import pandas as pd
import numpy as np
import pytest


def test_save_load(data_gen):
    data_gen.generate()
    model = muvi.MuVI(
        data_gen.ys,
        data_gen.w_masks,
        data_gen.x,
        device="cpu",
    )
    model.fit(
        batch_size=0,
        n_epochs=10,
        n_particles=1,
        learning_rate=0.01,
        optimizer="clipped",
        verbose=0,
        seed=0,
    )

    muvi.save(model, "test_save_load")
    loaded_model = muvi.load("test_save_load")

    for vn in model.view_names:
        assert (model.observations[vn] == loaded_model.observations[vn]).all()
        assert (model.prior_masks[vn] == loaded_model.prior_masks[vn]).all()
        assert (model.prior_scales[vn] == loaded_model.prior_scales[vn]).all()

    assert (model.covariates == loaded_model.covariates).all()
    assert model.likelihoods == loaded_model.likelihoods
    assert model.device == loaded_model.device
    assert model.reg_hs == loaded_model.reg_hs
    assert model._informed == loaded_model._informed
    assert model.nmf == loaded_model.nmf
    assert model._built == loaded_model._built
    assert model._trained == loaded_model._trained
    assert model._training_log == loaded_model._training_log
    assert model._cache == loaded_model._cache


def test_add_metadata():
    model = muvi.load("test_save_load")
    metadata = pd.Series(range(model.n_samples), index=model.sample_names)
    muvi.tl.add_metadata(model, "test", metadata)

    assert (muvi.tl.get_metadata(model, "test") == metadata).all()


def test_add_metadata_exist_error():
    model = muvi.load("test_save_load")
    metadata = pd.Series(range(model.n_samples), index=model.sample_names)
    muvi.tl.add_metadata(model, "test", metadata)
    with pytest.raises(ValueError):
        muvi.tl.add_metadata(model, "test", metadata)

    with pytest.raises(ValueError):
        muvi.tl.get_metadata(model, "test2")


def test_variance_explained():
    model = muvi.load("test_save_load")
    view_scores, factor_scores, cov_scores = muvi.tl.variance_explained(model)
    assert len(view_scores) == model.n_views
    assert factor_scores.shape == (model.n_factors, model.n_views)
    assert cov_scores.shape == (model.n_covariates, model.n_views)


def test_filter_factors():
    model = muvi.load("test_save_load")
    muvi.tl.variance_explained(model)
    muvi.tl.filter_factors(model, r2_thresh=0.95)
    assert "filtered_scores" in model._cache.factor_adata.obsm

    model._cache = None
    muvi.tl.variance_explained(model)
    muvi.tl.filter_factors(model, r2_thresh=5)
    assert model._cache.factor_adata.obsm["filtered_scores"].shape[1] == 5


def test_variance_explained_grouped():
    model = muvi.load("test_save_load")
    n_groups = 2
    metadata = pd.Series(
        [i // (model.n_samples // n_groups) for i in range(model.n_samples)],
        index=model.sample_names,
    )
    muvi.tl.add_metadata(model, "test", metadata)

    grouped_variance = muvi.tl.variance_explained_grouped(model, "test")
    assert grouped_variance.shape == (n_groups * model.n_factors, 2 + model.n_views)
    assert (
        grouped_variance.columns
        == ["test", "Factor"] + [f"r2_view_{m}" for m in range(model.n_views)]
    ).all()


def test_test():
    model = muvi.load("test_save_load")
    view_idx = "view_0"
    informed_factors = model.prior_masks[view_idx].any(axis=1)

    result = muvi.tl.test(model, min_size=1)

    assert "t" in result and "p" in result and "p_adj" in result
    assert (model.factor_names[informed_factors] == result["t"].columns).all()
    assert "t_all_view_0" in model._cache.factor_metadata
    assert "p_all_view_0" in model._cache.factor_metadata
    assert "p_adj_all_view_0" in model._cache.factor_metadata
    assert model._cache.factor_metadata["t_all_view_0"].isna()[~informed_factors].all()
    assert ~model._cache.factor_metadata["t_all_view_0"].isna()[informed_factors].all()


def test_from_adata(data_gen):
    data_gen.generate()

    adata = data_gen.to_mdata()["feature_group_0"]
    adata.obsm["x"] = data_gen.x
    model = muvi.tl.from_adata(
        adata,
        prior_mask_key="w_mask",
        covariate_key="x",
        device="cpu",
    )
    model.fit(
        batch_size=0,
        n_epochs=10,
        n_particles=1,
        learning_rate=0.01,
        optimizer="clipped",
        verbose=0,
        seed=0,
    )

    assert model._trained
    assert model._cache is None


def test_from_mdata(data_gen):
    data_gen.generate()

    model = muvi.tl.from_mdata(
        data_gen.to_mdata(),
        prior_mask_key="w_mask",
        covariate_key="x",
        device="cpu",
    )
    model.fit(
        batch_size=0,
        n_epochs=10,
        n_particles=1,
        learning_rate=0.01,
        optimizer="clipped",
        verbose=0,
        seed=0,
    )

    assert model._trained
    assert model._cache is None


def test_to_mdata():
    model = muvi.load("test_save_load")
    mdata = muvi.tl.to_mdata(model)

    assert mdata.shape == (model.n_samples, sum(model.n_features.values()))
    assert (mdata.obs_names == model.sample_names).all()

    np.testing.assert_equal(mdata.obsm["Z"], model.get_factor_scores(as_df=False))
    np.testing.assert_equal(mdata.obsm["X"], model.get_covariates(as_df=False))

    for vn in model.view_names:
        assert mdata[vn].shape == (model.n_samples, model.n_features[vn])
        assert (mdata[vn].var_names == model.feature_names[vn]).all()
        np.testing.assert_equal(
            mdata[vn].varm["W"], model.get_factor_loadings(as_df=False)[vn].T
        )
        np.testing.assert_equal(
            mdata[vn].varm["mask"], model.get_prior_masks(as_df=False)[vn].T
        )
        np.testing.assert_equal(
            mdata[vn].varm["B"], model.get_covariate_coefficients(as_df=False)[vn].T
        )
