import numpy as np
import pytest

from relife2.survival.data.measures import (
    MeasuresParserFrom1D,
    MeasuresParserFrom2D,
    intersect_measures,
)


@pytest.fixture
def example_1d_data():
    return {
        "observed_lifetimes": np.array([10, 11, 9, 10, 12, 13, 11]),
        "event": np.array([1, 0, 1, 0, 0, 0, 1]),
        "entry": np.array([0, 0, 3, 5, 3, 1, 9]),
    }


@pytest.fixture
def example_2d_data():
    return {
        "observed_lifetimes": np.array(
            [[1, 2], [0, 4], [5, 5], [7, np.inf], [10, 10], [2, 10], [10, 11]]
        ),
        "entry": np.array([0, 0, 3, 5, 3, 1, 9]),
        "departure": np.array([4, 0, 7, 10, 0, 12, 0]),
    }


def test_1d_data(example_1d_data):
    # data = Data(
    #     example_1d_data["observed_lifetimes"],
    #     right_censored_indicators=example_1d_data["event"] == 0,
    #     complete_indicators=example_1d_data["event"] == 1,
    #     entry=example_1d_data["entry"],
    # )
    parser = MeasuresParserFrom1D(
        example_1d_data["observed_lifetimes"],
        rc_indicators=1 - example_1d_data["event"],
        entry=example_1d_data["entry"],
    )

    (
        complete_lifetimes,
        left_censorships,
        right_censorships,
        interval_censorships,
        left_truncations,
        right_truncations,
    ) = parser()

    print(complete_lifetimes.unit_ids)
    assert np.all(complete_lifetimes.unit_ids == np.array([0, 2, 6]))
    assert np.all(complete_lifetimes.values == np.array([10, 9, 11]).reshape(-1, 1))

    assert np.all(right_censorships.unit_ids == np.array([1, 3, 4, 5]))
    assert np.all(right_censorships.values == np.array([11, 10, 12, 13]).reshape(-1, 1))

    assert np.all(left_truncations.unit_ids == np.array([2, 3, 4, 5, 6]))
    assert np.all(left_truncations.values == np.array([3, 5, 3, 1, 9]).reshape(-1, 1))

    assert np.all(
        intersect_measures(complete_lifetimes, left_truncations).values[:, [0]]
        == np.array([9, 11]).reshape(-1, 1)
    )

    assert np.all(
        intersect_measures(complete_lifetimes, left_truncations).values[:, [1]]
        == np.array([3, 9]).reshape(-1, 1)
    )

    assert np.all(
        intersect_measures(left_truncations, complete_lifetimes).values[:, [0]]
        == np.array([3, 9]).reshape(-1, 1)
    )

    assert np.all(
        intersect_measures(left_truncations, complete_lifetimes).values[:, [1]]
        == np.array([9, 11]).reshape(-1, 1)
    )


def test_2d_data(example_2d_data):
    parser = MeasuresParserFrom2D(
        example_2d_data["observed_lifetimes"],
        entry=example_2d_data["entry"],
        departure=example_2d_data["departure"],
    )

    (
        complete_lifetimes,
        left_censorships,
        right_censorships,
        interval_censorships,
        left_truncations,
        right_truncations,
    ) = parser()

    assert np.all(left_censorships.unit_ids == np.array([1]))
    assert np.all(left_censorships.values == np.array([4]).reshape(-1, 1))

    assert np.all(right_censorships.unit_ids == np.array([3]))
    assert np.all(right_censorships.values == np.array([7]).reshape(-1, 1))

    assert np.all(interval_censorships.unit_ids == np.array([0, 5, 6]))
    assert np.all(interval_censorships.values == np.array([[1, 2], [2, 10], [10, 11]]))

    # left truncations
    print(left_truncations)
    assert np.all(left_truncations.unit_ids == np.array([2, 3, 4, 5, 6]))
    assert np.all(left_truncations.values == np.array([3, 5, 3, 1, 9]).reshape(-1, 1))

    assert np.all(
        intersect_measures(left_truncations, interval_censorships).unit_ids
        == np.array([5, 6])
    )

    assert np.all(
        intersect_measures(left_truncations, interval_censorships).values[:, 1:]
        == np.array([[2, 10], [10, 11]])
    )

    assert np.all(
        intersect_measures(right_censorships, left_truncations).values[:, [0]]
        == np.array([7]).reshape(-1, 1)
    )
    assert np.all(
        intersect_measures(right_censorships, left_truncations).unit_ids
        == np.array([3])
    )
