import pytest
import numpy as np
import pandas as pd
import muvi


@pytest.mark.parametrize("nmf_flag", [False, True])
@pytest.mark.parametrize("use_informed", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("check_reconstruction", [False, True])
def test_save_load_roundtrip(
    data_gen, nmf_flag, use_informed, use_cache, check_reconstruction, tmp_path
):
    """
    Comprehensive roundtrip test for MuVI.save() and MuVI.load().
    Ensures:
        - all structural attributes restored
        - all observations + prior masks/scales restored
        - all guide variational params restored
        - all factor/order/sign restored
        - reconstruction identical (optional, expensive)
    """

    # -----------------------------
    # Generate synthetic test data
    # -----------------------------
    data_gen.generate()
    observations = data_gen.ys
    covs = data_gen.x

    # conditional informative prior
    if use_informed:
        prior_masks = data_gen.w_masks
        n_factors = None
    else:
        prior_masks = None
        n_factors = data_gen.n_factors

    # -----------------------------
    # Train model
    # -----------------------------
    model = muvi.MuVI(
        observations,
        prior_masks,
        covs,
        n_factors=n_factors,
        nmf=nmf_flag,
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

    if use_cache:
        muvi.tl.variance_explained(model)
        if use_informed:
            muvi.tl.test(model)

    model._reset_factors()

    # -----------------------------
    # Save + Load
    # -----------------------------
    save_dir = tmp_path
    model.save(save_dir)
    loaded = muvi.MuVI.load(save_dir, map_location="cpu")

    if not nmf_flag and use_informed and not use_cache:
        # save for the next test cases
        model.save("test_save_load")
        loaded = muvi.MuVI.load(save_dir, map_location="cpu")

    # ======================================================
    # 1) View / Sample / Feature name identities
    # ======================================================

    np.testing.assert_array_equal(model.view_names, loaded.view_names)
    np.testing.assert_array_equal(model.sample_names, loaded.sample_names)

    for vn in model.view_names:
        np.testing.assert_array_equal(model.feature_names[vn], loaded.feature_names[vn])

    # ======================================================
    # 2) Observations, Covariates, Prio Masks / Scales
    # ======================================================

    for vn in model.view_names:

        # Observations
        np.testing.assert_allclose(
            model.observations[vn],
            loaded.observations[vn],
            atol=1e-8,
            rtol=1e-6,
        )

        # Prior masks (exact boolean)
        if model.prior_masks is None:
            assert loaded.prior_masks is None
        else:
            np.testing.assert_array_equal(model.prior_masks[vn], loaded.prior_masks[vn])

        # Prior scales (float)
        if model.prior_scales is None:
            assert loaded.prior_scales is None
        else:
            np.testing.assert_allclose(
                model.prior_scales[vn],
                loaded.prior_scales[vn],
                atol=1e-8,
                rtol=1e-6,
            )

    # Covariates
    if model.covariates is None:
        assert loaded.covariates is None
    else:
        np.testing.assert_allclose(
            model.covariates,
            loaded.covariates,
            atol=1e-8,
            rtol=1e-6,
        )

    # ======================================================
    # 3) Metadata Fields
    # ======================================================

    assert model.likelihoods == loaded.likelihoods
    assert model.nmf == loaded.nmf
    assert model.reg_hs == loaded.reg_hs
    assert model._informed == loaded._informed
    assert model.n_factors == loaded.n_factors
    assert model.n_dense_factors == loaded.n_dense_factors
    assert model.n_covariates == loaded.n_covariates
    assert model.normalize == loaded.normalize
    assert model.prior_confidence == loaded.prior_confidence
    assert model.informed_views == loaded.informed_views

    # training state
    assert model._built == loaded._built
    assert model._trained == loaded._trained
    assert model._training_log == loaded._training_log

    # ======================================================
    # 4) Factor ordering, names, and signs
    # ======================================================

    np.testing.assert_array_equal(model.factor_names, loaded.factor_names)
    np.testing.assert_array_equal(model.factor_order, loaded.factor_order)
    np.testing.assert_allclose(model.factor_signs, loaded.factor_signs)

    # ======================================================
    # 5) Variational parameters (guide params)
    # ======================================================

    for (k1, p1), (k2, p2) in zip(
        model._guide.locs.named_parameters(), loaded._guide.locs.named_parameters()
    ):
        assert k1 == k2
        np.testing.assert_allclose(
            p1.detach().numpy(), p2.detach().numpy(), atol=1e-8, rtol=1e-6
        )

    for (k1, p1), (k2, p2) in zip(
        model._guide.scales.named_parameters(), loaded._guide.scales.named_parameters()
    ):
        assert k1 == k2
        np.testing.assert_allclose(
            p1.detach().numpy(), p2.detach().numpy(), atol=1e-8, rtol=1e-6
        )

    # ======================================================
    # 6) MuVIModel fields
    # ======================================================

    assert model._model.n_samples == loaded._model.n_samples
    assert model._model.n_factors == loaded._model.n_factors
    assert model._model.n_covariates == loaded._model.n_covariates
    assert model._model.n_views == loaded._model.n_views

    assert model._model.likelihoods == loaded._model.likelihoods
    assert model._model.nmf == loaded._model.nmf
    assert model._model.reg_hs == loaded._model.reg_hs

    # ======================================================
    # 7) Cached AnnData
    # ======================================================
    if model._cache is None:
        assert loaded._cache is None
    else:
        assert type(model._cache) is type(loaded._cache)

        # factor adata
        if model._cache.factor_adata is None:
            assert loaded._cache.factor_adata is None
        else:
            np.testing.assert_allclose(
                model._cache.factor_adata.X,
                loaded._cache.factor_adata.X,
                atol=1e-8,
                rtol=1e-6,
            )

        # cov adata
        if model._cache.cov_adata is None:
            assert loaded._cache.cov_adata is None
        else:
            np.testing.assert_allclose(
                model._cache.cov_adata.X,
                loaded._cache.cov_adata.X,
                atol=1e-8,
                rtol=1e-6,
            )

    # ======================================================
    # 8) Reconstructed values (optional)
    # ======================================================
    if check_reconstruction:
        for vn in model.view_names:
            rec1 = model.get_reconstructed(view_idx=vn, as_df=False)[vn]
            rec2 = loaded.get_reconstructed(view_idx=vn, as_df=False)[vn]
            np.testing.assert_allclose(rec1, rec2, atol=1e-8, rtol=1e-6)


def test_add_metadata():
    model = muvi.MuVI.load("test_save_load")
    metadata = pd.Series(range(model.n_samples), index=model.sample_names)
    muvi.tl.add_metadata(model, "test", metadata)

    assert (muvi.tl.get_metadata(model, "test") == metadata).all()


def test_add_metadata_exist_error():
    model = muvi.MuVI.load("test_save_load")
    metadata = pd.Series(range(model.n_samples), index=model.sample_names)
    muvi.tl.add_metadata(model, "test", metadata)
    with pytest.raises(ValueError):
        muvi.tl.add_metadata(model, "test", metadata)

    with pytest.raises(ValueError):
        muvi.tl.get_metadata(model, "test2")


def test_variance_explained():
    model = muvi.MuVI.load("test_save_load")
    view_scores, factor_scores, cov_scores = muvi.tl.variance_explained(model)
    assert len(view_scores) == model.n_views
    assert factor_scores.shape == (model.n_factors, model.n_views)
    assert cov_scores.shape == (model.n_covariates, model.n_views)


def test_filter_factors():
    model = muvi.MuVI.load("test_save_load")
    muvi.tl.variance_explained(model)
    muvi.tl.filter_factors(model, r2_thresh=0.95)
    assert "filtered_scores" in model._cache.factor_adata.obsm

    model._cache = None
    muvi.tl.variance_explained(model)
    muvi.tl.filter_factors(model, r2_thresh=5)
    assert model._cache.factor_adata.obsm["filtered_scores"].shape[1] == 5


def test_variance_explained_grouped():
    model = muvi.MuVI.load("test_save_load")
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
    model = muvi.MuVI.load("test_save_load")
    view_idx = "view_0"
    informed_factors = model.prior_masks[view_idx].any(axis=1)

    result = muvi.tl._test_single_view(model, min_size=1)

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
    model = muvi.MuVI.load("test_save_load")
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


def test_posterior_feature_sets():
    model = muvi.MuVI.load("test_save_load")
    posterior_feature_sets = muvi.tl.posterior_feature_sets(
        model, r2_thresh=1.0, dir_path="posterior_feature_sets"
    )
    assert len(posterior_feature_sets) == model.n_views
    for view_name in model.view_names:
        assert view_name in posterior_feature_sets
