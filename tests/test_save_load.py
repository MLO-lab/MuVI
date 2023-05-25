from muvi import MuVI, save, load


def test_save_load(data_gen):
    data_gen.generate()
    model = MuVI(
        data_gen.ys,
        data_gen.w_masks,
        data_gen.x,
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

    save(model, "test_save_load")
    loaded_model = load("test_save_load")

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
