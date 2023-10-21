from muvi import MuVI


def test_fit(data_gen):
    data_gen.generate()
    model = MuVI(
        data_gen.ys,
        data_gen.w_masks,
        data_gen.x,
        device="cpu",
    )
    model.fit(
        batch_size=0,
        n_epochs=10,
        n_particles=5,
        learning_rate=0.01,
        optimizer="clipped",
        verbose=0,
        seed=0,
    )

    assert model._trained
    assert model._cache is None


def test_fit_minibatch(data_gen):
    data_gen.generate()
    model = MuVI(
        data_gen.ys,
        data_gen.w_masks,
        data_gen.x,
        device="cpu",
    )
    model.fit(batch_size=model.n_samples // 2, n_epochs=1)

    assert model._trained
    assert model._cache is None
