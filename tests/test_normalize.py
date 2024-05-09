import numpy as np

from muvi import MuVI


def test_normalize(pandas_input):

    observations_df = pandas_input["observations"]
    observations_array = [obs.to_numpy() for obs in observations_df]

    for observations in [observations_array, observations_df]:
        for m in range(len(observations)):
            observations[m] = observations[m] * 5 + 5

        observations_centered = MuVI(observations, n_factors=5).observations
        observations_min_to_zero = MuVI(
            observations, n_factors=5, nmf=True
        ).observations

        for view_name in observations_centered.keys():
            n_feat = observations_centered[view_name].shape[1]
            np.testing.assert_allclose(
                np.nanmean(observations_centered[view_name], axis=0),
                np.zeros(n_feat),
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.nanmin(observations_min_to_zero[view_name], axis=0),
                np.zeros(n_feat),
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.nanstd(observations_centered[view_name]),
                np.ones(1),
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.nanstd(observations_min_to_zero[view_name]),
                np.ones(1),
                rtol=1e-5,
                atol=1e-5,
            )
