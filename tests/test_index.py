import numpy as np

from muvi import MuVI


def test_get_observations_view(data_gen):
    data_gen.generate()
    model = MuVI(
        data_gen.ys,
        data_gen.w_masks,
        data_gen.x,
        use_gpu=False,
    )

    # single view
    for view_idx in [0, [0], "view_0", ["view_0"]]:
        np.testing.assert_array_equal(
            data_gen.ys[0],
            model.get_observations(view_idx)["view_0"],
        )

    # multiple views
    view_indices = [2, 1]
    view_names = [f"view_{m}" for m in view_indices]
    true_obs = np.concatenate([data_gen.ys[m] for m in view_indices], 1)
    for view_idx in [view_indices, view_names]:
        obs = model.get_observations(view_indices)
        obs = np.concatenate([obs[vn] for vn in view_names], axis=1)
        np.testing.assert_array_equal(true_obs, obs)


def test_get_observations_sample(data_gen):
    data_gen.generate()
    model = MuVI(
        data_gen.ys,
        data_gen.w_masks,
        data_gen.x,
        use_gpu=False,
    )

    for sample_idx in [5, [5], "sample_5", ["sample_5"]]:
        np.testing.assert_array_equal(
            data_gen.ys[1][5, :],
            model.get_observations("view_1", sample_idx=sample_idx)["view_1"][0, :],
        )


def test_get_observations_feature(data_gen):
    data_gen.generate()
    model = MuVI(
        data_gen.ys,
        data_gen.w_masks,
        data_gen.x,
        use_gpu=False,
    )

    # single view
    for view_idx in [1, [1], "view_1", ["view_1"]]:
        for feature_idx in [
            5,
            [5],
            "view_1_feature_5",
            ["view_1_feature_5"],
            {1: 5},
            {1: "view_1_feature_5"},
            {"view_1": 5},
            {"view_1": "view_1_feature_5"},
        ]:
            np.testing.assert_array_equal(
                data_gen.ys[1][:, 5],
                model.get_observations(view_idx, feature_idx=feature_idx)["view_1"][
                    :, 0
                ],
            )

    # multiple views
    view_indices = [2, 1]
    view_names = [f"view_{m}" for m in view_indices]
    true_obs = np.concatenate([data_gen.ys[m] for m in view_indices], 1)[
        :, [5, data_gen.n_features[view_indices[0]] + 5]
    ]

    feature_idx_as_list = [
        [5, 5],
        [[5], [5]],
        [[5], "view_1_feature_5"],
        [["view_2_feature_5"], 5],
        ["view_2_feature_5", "view_1_feature_5"],
        [["view_2_feature_5"], ["view_1_feature_5"]],
    ]

    feature_idx_as_dict = [
        {2: 5, 1: 5},
        {2: 5, "view_1": 5},
        {2: [5], "view_1": [5]},
        {2: "view_2_feature_5", "view_1": "view_1_feature_5"},
        {2: ["view_2_feature_5"], "view_1": ["view_1_feature_5"]},
    ]

    for view_idx in [view_indices, view_names]:
        for feature_idx in feature_idx_as_list + feature_idx_as_dict:
            obs = model.get_observations(view_indices, feature_idx=feature_idx)
            obs = np.concatenate([obs[vn] for vn in view_names], axis=1)
            np.testing.assert_array_equal(true_obs, obs)

    for feature_idx in feature_idx_as_dict:
        obs = model.get_observations(None, feature_idx=feature_idx)
        obs = np.concatenate([obs[vn] for vn in view_names], axis=1)
        np.testing.assert_array_equal(true_obs, obs)
