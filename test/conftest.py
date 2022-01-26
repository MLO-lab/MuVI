import pytest

from spexlvm import data


@pytest.fixture(scope="function")
def data_generator():
    return data.DataGenerator(
        n_samples=200,
        n_features=2000,
        n_factors=100,
        likelihood="normal",
        factor_size_params=(0.05, 0.15),
        factor_size_dist="uniform",
        n_active_factors=1.0,
    )
