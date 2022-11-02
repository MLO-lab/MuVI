from muvi import MuVI


def test_fewer_n_factors(pandas_input):
    model = MuVI(
        pandas_input["observations"],
        pandas_input["masks"],
        pandas_input["covariates"],
        n_factors=pandas_input["n_factors"] - 2,
        view_names=pandas_input["view_names"],
        use_gpu=False,
    )

    assert model.n_factors == pandas_input["n_factors"]


def test_more_n_factors(pandas_input):

    n_dense = 2

    model = MuVI(
        pandas_input["observations"],
        pandas_input["masks"],
        pandas_input["covariates"],
        n_factors=pandas_input["n_factors"] + n_dense,
        view_names=pandas_input["view_names"],
        use_gpu=False,
    )

    assert model.n_factors == pandas_input["n_factors"] + n_dense
    assert (
        model.factor_names[-n_dense:] == [f"dense_{k}" for k in range(n_dense)]
    ).all()

    for prior_mask in model.get_prior_masks().values():
        assert prior_mask.shape[0] == model.n_factors
        assert prior_mask[-n_dense:, :].all()
