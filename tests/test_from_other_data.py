import muvi


def test_from_adata(data_gen):
    data_gen.generate()

    adata = data_gen.to_mdata()["feature_group_0"]
    adata.obsm["x"] = data_gen.x
    model = muvi.tl.from_adata(
        adata,
        prior_mask_key="w_mask",
        covariate_key="x",
        use_gpu=False,
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
        use_gpu=False,
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
