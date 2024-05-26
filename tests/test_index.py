import numpy as np
import pandas as pd
import pytest

from muvi import MuVI
from muvi.core.index import _normalize_index, _make_index_unique


def test_normalize_index_correct():
    index = pd.Index([str(i) for i in range(10)])
    single_indices = [0, "0", True]
    multiple_indices_int = list(range(3))
    multiple_indices_str = [str(i) for i in multiple_indices_int]
    multiple_indices_bool = [True for _ in multiple_indices_int] + [False]
    multiple_indices = [
        multiple_indices_int,
        multiple_indices_str,
        multiple_indices_bool,
    ]

    assert (_normalize_index("all", index) == np.arange(len(index))).all()
    assert (_normalize_index("all", index, as_idx=False) == np.array(index)).all()

    for si in single_indices:
        for indexer in [si, [si], np.array([si]), pd.Index([si])]:
            assert _normalize_index(indexer, index) == np.array([single_indices[0]])
            assert _normalize_index(indexer, index, as_idx=False) == np.array(
                [single_indices[1]]
            )

    for mi in multiple_indices:
        for indexer in [mi, np.array(mi), pd.Index(mi)]:
            assert (
                _normalize_index(indexer, index) == np.array(multiple_indices[0])
            ).all()
            assert (
                _normalize_index(indexer, index, as_idx=False)
                == np.array(multiple_indices[1])
            ).all()


def test_normalize_index_incorrect():
    index = pd.Index([str(i) for i in range(10)])

    with pytest.raises(IndexError, match="Empty index"):
        _normalize_index([], index)

    with pytest.raises(IndexError, match="Empty index"):
        _normalize_index([False], index)


def test_make_index_unique():
    index = pd.Index([str(i) for i in range(10)] + ["0"])
    deduped_index = _make_index_unique(index)
    assert deduped_index[-1] == "0_1"


def test_get_observations_view(data_gen):
    data_gen.generate()
    model = MuVI(
        data_gen.ys,
        data_gen.w_masks,
        data_gen.x,
        normalize=False,
        device="cpu",
    )

    # single view
    for view_idx in [0, [0], "view_0", ["view_0"]]:
        np.testing.assert_allclose(
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
        np.testing.assert_allclose(true_obs, obs)


def test_get_observations_sample(data_gen):
    data_gen.generate()
    model = MuVI(
        data_gen.ys,
        data_gen.w_masks,
        data_gen.x,
        normalize=False,
        device="cpu",
    )

    for sample_idx in [5, [5], "sample_5", ["sample_5"]]:
        np.testing.assert_allclose(
            data_gen.ys[1][5, :],
            model.get_observations("view_1", sample_idx=sample_idx)["view_1"][0, :],
        )


def test_get_observations_feature(data_gen):
    data_gen.generate()
    model = MuVI(
        data_gen.ys,
        data_gen.w_masks,
        data_gen.x,
        normalize=False,
        device="cpu",
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
            np.testing.assert_allclose(
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
            np.testing.assert_allclose(true_obs, obs)

    for feature_idx in feature_idx_as_dict:
        obs = model.get_observations(None, feature_idx=feature_idx)
        obs = np.concatenate([obs[vn] for vn in view_names], axis=1)
        np.testing.assert_allclose(true_obs, obs)
