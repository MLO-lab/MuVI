import numpy as np
import pytest
from pandas.api.types import is_string_dtype

from muvi import MuVI


def test_numpy_input(data_gen):
    data_gen.generate()
    model = MuVI(
        data_gen.ys,
        data_gen.w_masks,
        data_gen.x,
        device="cpu",
    )

    assert model.n_views == data_gen.n_views
    assert model.n_samples == data_gen.n_samples
    assert model.n_covariates == data_gen.n_covariates
    assert model.n_factors == data_gen.n_factors
    for m, vn in enumerate(model.view_names):
        assert model.n_features[vn] == data_gen.n_features[m]

    assert is_string_dtype(model.view_names)
    assert is_string_dtype(model.sample_names)
    assert is_string_dtype(model.covariate_names)
    assert is_string_dtype(model.factor_names)
    for vn in model.view_names:
        assert is_string_dtype(model.feature_names[vn])

    assert model.likelihoods == {vn: "normal" for vn in model.view_names}

    model._setup_training_data()


def test_pandas_input(pandas_input):
    model = MuVI(
        pandas_input["observations"],
        pandas_input["masks"],
        pandas_input["covariates"],
        view_names=pandas_input["view_names"],
        device="cpu",
    )

    assert all(model.view_names == pandas_input["view_names"])
    assert all(model.sample_names == pandas_input["observations"][0].index)
    assert all(model.covariate_names == pandas_input["covariates"].columns)
    assert all(model.factor_names == pandas_input["masks"][-1].index)
    for m, vn in enumerate(model.view_names):
        all(model.feature_names[vn] == pandas_input["observations"][m].columns)

    model._setup_training_data()


def test_pandas_input_shuffled_samples(pandas_input):
    observations = pandas_input["observations"]
    sample_names = observations[0].index.tolist()
    np.random.shuffle(sample_names)
    observations[1] = observations[1].loc[sample_names, :]

    model = MuVI(
        observations,
        pandas_input["masks"],
        pandas_input["covariates"],
        view_names=pandas_input["view_names"],
        device="cpu",
    )

    assert all(model.sample_names == observations[0].index)

    model._setup_training_data()


def test_pandas_input_missing_cov_samples(pandas_input):
    with pytest.raises(ValueError, match="Covariates have fewer samples than expected"):
        MuVI(
            pandas_input["observations"],
            covariates=pandas_input["covariates"].iloc[:-1, :],
            n_factors=pandas_input["n_factors"],
            device="cpu",
        )


def test_pandas_input_missing_some_mask_features(pandas_input):
    bad_masks = pandas_input["masks"]
    bad_masks[0] = bad_masks[0].iloc[:, :-1]
    model = MuVI(
        pandas_input["observations"],
        bad_masks,
        pandas_input["covariates"],
        device="cpu",
    )

    assert (
        model.feature_names["view_0"]
        == model.get_prior_masks(as_df=True)["view_0"].columns
    ).all()


def test_pandas_input_missing_all_mask_features(pandas_input):
    bad_masks = pandas_input["masks"]
    bad_masks[0].columns = bad_masks[0].columns.astype(str) + "_suffix"

    with pytest.raises(ValueError, match="None of the feature names for mask"):
        MuVI(
            pandas_input["observations"],
            bad_masks,
            pandas_input["covariates"],
            device="cpu",
        )
