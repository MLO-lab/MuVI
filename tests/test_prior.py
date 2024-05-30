from muvi import MuVI


def test_more_n_factors(pandas_input):
    n_factors = 2

    model = MuVI(
        pandas_input["observations"],
        pandas_input["masks"],
        pandas_input["covariates"],
        n_factors=n_factors,
        view_names=pandas_input["view_names"],
        device="cpu",
    )

    assert model.n_factors == pandas_input["n_factors"] + n_factors
    assert (
        model.factor_names[-n_factors:] == [f"dense_{k}" for k in range(n_factors)]
    ).all()

    for prior_mask in model.get_prior_masks().values():
        assert prior_mask.shape[0] == model.n_factors
        assert prior_mask[-n_factors:, :].all()
