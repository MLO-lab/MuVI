import pandas as pd
import pytest

from muvi import DataGenerator


@pytest.fixture(scope="function")
def data_gen():
    return DataGenerator(
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


@pytest.fixture(scope="function")
def pandas_input(data_gen):
    data_gen.generate()
    view_names = [f"view_{m}" for m in range(data_gen.n_views)]
    sample_names = [f"sample_{i}" for i in range(data_gen.n_samples)]
    covariate_names = [f"covariate_{k}" for k in range(data_gen.n_covariates)]
    factor_names = [f"factor_{k}" for k in range(data_gen.n_factors)]
    feature_names = [
        [f"view_{m}_feature_{j}" for j in range(data_gen.n_features[m])]
        for m in range(data_gen.n_views)
    ]

    return {
        "view_names": view_names,
        "n_factors": data_gen.n_factors,
        "observations": [
            pd.DataFrame(y, index=sample_names, columns=feature_names[m])
            for m, y in enumerate(data_gen.ys)
        ],
        "covariates": pd.DataFrame(
            data_gen.x, index=sample_names, columns=covariate_names
        ),
        "masks": [
            pd.DataFrame(mask, index=factor_names, columns=feature_names[m])
            for m, mask in enumerate(data_gen.w_masks)
        ],
    }
