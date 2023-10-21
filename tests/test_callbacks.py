from muvi import MuVI
from muvi import EarlyStoppingCallback, LogCallback


def test_early_stopping_callback(data_gen):
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
        callbacks=[EarlyStoppingCallback(10, min_epochs=1, tolerance=100, patience=1)],
    )

    assert model._training_log["stop_early"]


def test_log_callback(data_gen):
    data_gen.generate()
    model = MuVI(
        data_gen.ys,
        data_gen.w_masks,
        data_gen.x,
        device="cpu",
    )

    n_epochs = 9
    log_callback = LogCallback(
        model,
        n_epochs,
        n_checkpoints=3,
        masks={
            vn: data_gen.w_masks[m].astype(bool)
            for m, vn in enumerate(model.view_names)
        },
        binary_scores_at=500,
        threshold=0.1,
        log=True,
        n_annotated=model.n_factors,
    )

    model.fit(
        batch_size=0,
        n_epochs=10,
        n_particles=5,
        learning_rate=0.01,
        optimizer="clipped",
        verbose=0,
        seed=0,
        callbacks=[log_callback],
    )

    assert len(log_callback.scores["sparsity_view_0"]) == 3
