import numpy as np

from muvi.tools.feature_sets import FeatureSet, FeatureSets


def test_redundant_features():
    feature_set = FeatureSet(name="redundant", features=list("AABBCC"))
    assert len(feature_set) == 3
    assert feature_set.redundant_features == {"A": 2, "B": 2, "C": 2}


def test_equal():
    feature_set_0 = FeatureSet(features=list("ABC"), name="feature_set_0")
    feature_set_1 = FeatureSet(features=list("ABC"), name="feature_set_1")
    feature_set_2 = FeatureSet(features=list("ABCD"), name="feature_set_2")
    assert feature_set_0 == feature_set_1
    assert feature_set_0 != feature_set_2

    feature_sets_0 = FeatureSets([feature_set_0, feature_set_1], name="feature_sets_0")
    feature_sets_1 = FeatureSets([feature_set_0, feature_set_1], name="feature_sets_1")
    feature_sets_2 = FeatureSets([feature_set_0, feature_set_2], name="feature_sets_2")

    assert feature_sets_0 == feature_sets_1
    assert feature_sets_0 != feature_sets_2


def test_union():
    feature_set_0 = FeatureSet(features=list("ABC"), name="feature_set_0")
    feature_set_1 = FeatureSet(features=list("CDE"), name="feature_set_1")

    assert (feature_set_0 | feature_set_1) == FeatureSet(
        features=list("ABCDE"), name="feature_set_0|feature_set_1"
    )

    feature_sets_0 = FeatureSets([feature_set_0], name="feature_sets_0")
    feature_sets_1 = FeatureSets([feature_set_1], name="feature_sets_1")

    assert (feature_sets_0 | feature_sets_1) == FeatureSets(
        [feature_set_0, feature_set_1], name="feature_sets_0|feature_sets_1"
    )


def test_intersection():
    feature_set_0 = FeatureSet(features=list("ABC"), name="feature_set_0")
    feature_set_1 = FeatureSet(features=list("CDE"), name="feature_set_1")

    assert (feature_set_0 & feature_set_1) == FeatureSet(
        features=list("C"), name="feature_set_0&feature_set_1"
    )

    feature_sets_0 = FeatureSets([feature_set_0], name="feature_sets_0")
    feature_sets_1 = FeatureSets([feature_set_1], name="feature_sets_1")
    feature_sets_2 = FeatureSets([feature_set_0], name="feature_sets_2")

    assert (feature_sets_0 & feature_sets_1) == FeatureSets(
        [], name="feature_sets_0&feature_sets_1"
    )
    assert (feature_sets_0 & feature_sets_2) == FeatureSets(
        [feature_set_0], name="feature_sets_0&feature_sets_2"
    )


def test_find():
    feature_set_0 = FeatureSet(features=list("ABC"), name="feature_set_0")
    feature_set_1 = FeatureSet(features=list("CDE"), name="feature_set_1")

    feature_sets = FeatureSets([feature_set_0, feature_set_1], name="feature_sets")

    single_result = feature_sets.find("set_0")
    multiple_result = feature_sets.find("set")

    assert single_result == FeatureSets([feature_set_0], name="set_0")
    assert single_result.name == "feature_sets:set_0"
    assert multiple_result == feature_sets
    assert multiple_result.name == "feature_sets:set"


def test_remove_keep():
    feature_set_0 = FeatureSet(features=list("ABC"), name="feature_set_0")
    feature_set_1 = FeatureSet(features=list("CDE"), name="feature_set_1")

    feature_sets = FeatureSets([feature_set_0, feature_set_1], name="feature_sets")

    assert feature_sets.remove(["feature_set_0"]) == FeatureSets(
        [feature_set_1], name="feature_sets"
    )
    assert feature_sets.keep(["feature_set_0"]) == FeatureSets(
        [feature_set_0], name="feature_sets"
    )


def test_trim():
    feature_set_0 = FeatureSet(features=list("ABCDEFG"), name="feature_set_0")
    feature_set_1 = FeatureSet(features=list("XYZ"), name="feature_set_1")

    feature_sets = FeatureSets([feature_set_0, feature_set_1], name="feature_sets")
    assert feature_sets.trim(4).features == set(list("ABCDEFG"))
    assert feature_sets.trim(1, 4).features == set(list("XYZ"))


def test_subset():
    feature_set_0 = FeatureSet(features=list("ABCDEFG"), name="feature_set_0")
    feature_set_1 = FeatureSet(features=list("XYZ"), name="feature_set_1")

    feature_sets = FeatureSets([feature_set_0, feature_set_1], name="feature_sets")
    feature_sets = feature_sets.subset(list("ABC"))

    assert len(feature_sets) == 1
    assert feature_sets.features == set(list("ABC"))


def test_filter():
    feature_set_0 = FeatureSet(features=list("ABC"), name="feature_set_0")
    feature_set_1 = FeatureSet(features=list("CDE"), name="feature_set_1")
    feature_set_2 = FeatureSet(features=list("XYZ"), name="feature_set_2")

    feature_sets = FeatureSets(
        [feature_set_0, feature_set_1, feature_set_2], name="feature_sets"
    )

    assert feature_sets.filter(list("ABCDEX"), 0.1, 1).feature_sets == set(
        [
            feature_set_0,
            feature_set_1,
            FeatureSet(features=list("X"), name="feature_set_2"),
        ]
    )
    assert feature_sets.filter(list("ABCDEX"), 0.5, 1).feature_sets == set(
        [feature_set_0, feature_set_1]
    )
    assert feature_sets.filter(
        list("ABCDEX"), 0.1, 1, subset=False
    ).feature_sets == set([feature_set_0, feature_set_1, feature_set_2])
    assert feature_sets.filter(
        list("ABCDEX"),
        0.5,
        1,
        keep=["feature_set_2"],
        subset=False,
    ).feature_sets == set([feature_set_0, feature_set_1, feature_set_2])


def test_to_mask():
    feature_set_0 = FeatureSet(features=list("ABC"), name="feature_set_0")
    feature_set_1 = FeatureSet(features=list("CDE"), name="feature_set_1")
    feature_set_2 = FeatureSet(features=list("XYZ"), name="feature_set_2")

    feature_sets = FeatureSets(
        [feature_set_0, feature_set_1, feature_set_2], name="feature_sets"
    )

    mask = feature_sets.to_mask(sorted(feature_sets.features))

    assert (mask.index == sorted([fs.name for fs in feature_sets.feature_sets])).all()
    assert (mask.columns == sorted(feature_sets.features)).all()
    assert (
        mask.loc["feature_set_0", :]
        == [True, True, True, False, False, False, False, False]
    ).all()

    assert (
        mask.loc["feature_set_1", :]
        == [False, False, True, True, True, False, False, False]
    ).all()


def test_distance():
    feature_set_0 = FeatureSet(features=list("ABC"), name="feature_set_0")
    feature_set_1 = FeatureSet(features=list("CDE"), name="feature_set_1")
    feature_set_2 = FeatureSet(features=list("DEX"), name="feature_set_2")

    feature_sets = FeatureSets(
        [feature_set_0, feature_set_1, feature_set_2], name="feature_sets"
    )

    feature_sets_distance = feature_sets.distance()
    assert np.diag(feature_sets_distance.values).sum() == 0
    assert feature_sets_distance.loc["feature_set_0", "feature_set_1"] == 4 / 5
    assert feature_sets_distance.loc["feature_set_0", "feature_set_2"] == 1
    assert feature_sets_distance.loc["feature_set_1", "feature_set_2"] == 1 / 2


def test_find_similar_pairs():
    feature_set_0 = FeatureSet(features=list("ABC"), name="feature_set_0")
    feature_set_1 = FeatureSet(features=list("CDE"), name="feature_set_1")
    feature_set_2 = FeatureSet(features=list("DEX"), name="feature_set_2")

    feature_sets = FeatureSets(
        [feature_set_0, feature_set_1, feature_set_2], name="feature_sets"
    )

    assert feature_sets.find_similar_pairs(distance_threshold=0.8) == set(
        [("feature_set_0", "feature_set_1", 0.8)]
    )
    # 0.5 closer than 0.8
    assert feature_sets.find_similar_pairs(distance_threshold=0.5) == set(
        [("feature_set_1", "feature_set_2", 0.5)]
    )


def test_merge_similar():
    feature_set_0 = FeatureSet(features=list("ABC"), name="feature_set_0")
    feature_set_1 = FeatureSet(features=list("CDE"), name="feature_set_1")
    feature_set_2 = FeatureSet(features=list("DEX"), name="feature_set_2")

    feature_sets = FeatureSets(
        [feature_set_0, feature_set_1, feature_set_2], name="feature_sets"
    )

    assert feature_sets.merge_similar(distance_threshold=0.8) == FeatureSets(
        [
            FeatureSet(
                features=list("ABCDEX"),
                name="feature_set_0|feature_set_1|feature_set_2",
            )
        ]
    )

    assert feature_sets.merge_similar(distance_threshold=0.5) == FeatureSets(
        [
            FeatureSet(features=list("ABC"), name="feature_set_0"),
            FeatureSet(features=list("CDEX"), name="feature_set_1|feature_set_2"),
        ]
    )
