import numpy as np


def test_shapes(data_gen):
    data_gen.generate()

    assert data_gen.x.shape == (
        data_gen.n_samples,
        data_gen.n_covariates,
    )

    assert data_gen.z.shape == (
        data_gen.n_samples,
        data_gen.n_factors,
    )

    for m in range(data_gen.n_views):
        assert data_gen.betas[m].shape == (
            data_gen.n_covariates,
            data_gen.n_features[m],
        )

        assert data_gen.ws[m].shape == (
            data_gen.n_factors,
            data_gen.n_features[m],
        )

        assert len(data_gen.sigmas[m]) == data_gen.n_features[m]

        assert data_gen.ys[m].shape == (
            data_gen.n_samples,
            data_gen.n_features[m],
        )


def test_w_mask(data_gen):
    data_gen.generate()

    for m in range(data_gen.n_views):
        w = data_gen.ws[m]
        w_mask = data_gen.w_masks[m]
        assert w.shape == w_mask.shape
        assert ((np.abs(w) > 0.05) == w_mask).all()


def test_noisy_w_mask(data_gen):
    data_gen.generate()

    noise_fraction = 0.1
    data_gen.get_noisy_mask(noise_fraction=noise_fraction)

    for m in range(data_gen.n_views):
        w_mask = data_gen.w_masks[m].astype(bool)
        noisy_w_mask = data_gen.noisy_w_masks[m].astype(bool)

        p = w_mask.sum(1)
        tp = (noisy_w_mask & w_mask).sum(1)
        tpr = np.nanmean(tp / p)
        assert tpr > 0.85
        # assert tpr < 0.95


def test_missing_y_random(data_gen):
    data_gen.generate()

    random_fraction = 0.2

    data_gen.generate_missingness(
        random_fraction,
    )

    for m in range(data_gen.n_views):
        y = data_gen.ys[m]
        missing_y = data_gen.missing_ys[m]
        presence_mask = data_gen.presence_masks[m]
        assert y.shape == missing_y.shape
        assert y.shape == presence_mask.shape
        assert (~np.isnan(missing_y) == presence_mask).all()

        fraction_missing = 1 - presence_mask.mean()
        assert fraction_missing > random_fraction * 0.75
        assert fraction_missing < random_fraction * 1.25


def test_missing_y_partial_samples(data_gen):
    data_gen.generate()

    n_partial_samples = 10

    data_gen.generate_missingness(
        0.0,
        n_partial_samples,
        0,
        0.0,
    )

    missing_y = data_gen.y_to_matrix(missing=True)
    assert n_partial_samples == (np.isnan(missing_y).mean(1) > 0.0).sum()


def test_missing_y_partial_features(data_gen):
    data_gen.generate()

    n_partial_features = 50
    missing_fraction_partial_features = 0.2

    data_gen.generate_missingness(
        0.0,
        0,
        n_partial_features,
        missing_fraction_partial_features,
    )

    missing_y = data_gen.y_to_matrix(missing=True)
    assert (
        n_partial_features
        == (np.isnan(missing_y).mean(0) == missing_fraction_partial_features).sum()
    )
