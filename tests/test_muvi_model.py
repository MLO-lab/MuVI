import pytest
from muvi import MuVI
import numpy as np
import pandas as pd


def test_numpy_input(data_gen):
    data_gen.generate()
    model = MuVI(
        n_factors=data_gen.n_factors,
        observations=data_gen.ys,
        covariates=data_gen.x,
        use_gpu=False,
    )
    model.add_prior_masks(data_gen.w_masks)

    assert model.n_views == data_gen.n_views
    assert model.view_names == list(range(model.n_views))

    assert model.n_samples == data_gen.n_samples
    assert model.sample_names == list(range(model.n_samples))

    assert model.n_covariates == data_gen.n_covariates
    assert model.covariate_names == list(range(model.n_covariates))

    assert model.n_factors == data_gen.n_factors
    assert model.factor_names == list(range(model.n_factors))

    assert model.n_features == data_gen.n_features
    assert model.feature_names == [
        list(range(model.n_features[m])) for m in range(model.n_views)
    ]

    assert model.likelihoods == ["normal" for _ in range(model.n_views)]

    model._setup_training_data()


def test_pandas_input(pandas_input):
    model = MuVI(
        n_factors=pandas_input["n_factors"],
        observations=pandas_input["observations"],
        view_names=pandas_input["view_names"],
        covariates=pandas_input["covariates"],
        use_gpu=False,
    )
    model.add_prior_masks(pandas_input["masks"])

    assert model.view_names == pandas_input["view_names"]
    assert model.sample_names == pandas_input["observations"][0].index.tolist()
    assert model.covariate_names == pandas_input["covariates"].columns.tolist()
    assert model.factor_names == pandas_input["masks"][-1].index.tolist()
    assert model.feature_names == [
        pandas_input["observations"][m].columns.tolist() for m in range(model.n_views)
    ]

    model._setup_training_data()


def test_pandas_input_shuffled_samples(pandas_input):

    observations = pandas_input["observations"]
    sample_names = observations[0].index.tolist()
    np.random.shuffle(sample_names)
    observations[1] = observations[1].loc[sample_names, :]

    model = MuVI(
        n_factors=pandas_input["n_factors"],
        observations=observations,
        view_names=pandas_input["view_names"],
        covariates=pandas_input["covariates"],
        use_gpu=False,
    )
    model.add_prior_masks(pandas_input["masks"])

    assert model.sample_names == observations[0].index.tolist()

    model._setup_training_data()


def test_pandas_input_missing_samples(pandas_input):

    bad_observations = [obs.copy() for obs in pandas_input["observations"]]
    sample_names = bad_observations[0].index.tolist()[:-1]
    bad_observations[1] = bad_observations[1].loc[sample_names, :]

    with pytest.raises(
        ValueError, match="all views must have the same number of samples"
    ):
        model = MuVI(
            n_factors=pandas_input["n_factors"],
            observations=bad_observations,
            view_names=pandas_input["view_names"],
            covariates=pandas_input["covariates"],
            use_gpu=False,
        )

    with pytest.raises(
        ValueError, match="does not match the number of samples for the covariates"
    ):
        model = MuVI(
            n_factors=pandas_input["n_factors"],
            observations=pandas_input["observations"],
            view_names=pandas_input["view_names"],
            covariates=pandas_input["covariates"].iloc[:-1, :],
            use_gpu=False,
        )

    with pytest.raises(
        ValueError, match="all masks must have the same number of factors"
    ):
        model = MuVI(
            n_factors=pandas_input["n_factors"],
            observations=pandas_input["observations"],
            view_names=pandas_input["view_names"],
            covariates=pandas_input["covariates"],
            use_gpu=False,
        )
        masks = pandas_input["masks"]
        masks[0] = masks[0].iloc[:-1, :]
        model.add_prior_masks(masks)


def test_pandas_input_missing_features(pandas_input):

    model = MuVI(
        n_factors=pandas_input["n_factors"],
        observations=pandas_input["observations"],
        view_names=pandas_input["view_names"],
        covariates=pandas_input["covariates"],
        use_gpu=False,
    )
    masks = pandas_input["masks"]
    masks[0] = masks[0].iloc[:, :-1]
    with pytest.raises(
        ValueError, match="each mask must match the number of features of its view."
    ):
        model.add_prior_masks(masks)
