import pytest

from spexlvm import data


@pytest.fixture(scope="function")
def data_gen():
    return data.DataGenerator(
        n_samples=200,
        n_features=[400, 300, 200, 100],
        likelihoods=["normal" for _ in range(4)],
        n_fully_shared_factors=2,
        n_partially_shared_factors=14,
        n_private_factors=4,
        n_covariates=2,
        factor_size_params=(0.05, 0.15),
        factor_size_dist="uniform",
        n_active_factors=1.0,
    )
