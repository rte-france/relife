import numpy as np
import pytest

from relife.utils.data import (
    Lifetime1DParser,
    Lifetime2DParser,
    LifetimeData,
)


@pytest.fixture
def example_1d_data():
    return {
        "observed_lifetimes": np.array([10, 11, 9, 10, 12, 13, 11])
        .astype(np.float64)
        .reshape(-1, 1),
        "event": np.array([1, 0, 1, 0, 0, 0, 1]).astype(np.bool_).reshape(-1, 1),
        "entry": np.array([0, 0, 3, 5, 3, 1, 9]).astype(np.float64).reshape(-1, 1),
    }


@pytest.fixture
def example_2d_data():
    return {
        "observed_lifetimes": np.array(
            [[1, 2], [0, 4], [5, 5], [7, np.inf], [10, 10], [2, 10], [10, 11]]
        ).astype(np.float64),
        "entry": np.array([0, 0, 3, 5, 3, 1, 9]).astype(np.float64).reshape(-1, 1),
        "departure": np.array([4, np.inf, 7, 10, np.inf, 12, np.inf])
        .astype(np.float64)
        .reshape(-1, 1),
    }


def test_1d_data(example_1d_data):
    reader = Lifetime1DParser(
        example_1d_data["observed_lifetimes"],
        event=example_1d_data["event"],
        entry=example_1d_data["entry"],
    )

    lifetime_data = LifetimeData(
        len(example_1d_data),
        reader.get_complete(),
        reader.get_left_censoring(),
        reader.get_right_censoring(),
        reader.get_interval_censoring(),
        reader.get_left_truncation(),
        reader.get_right_truncation(),
    )

    assert np.all(lifetime_data.complete.index == np.array([0, 2, 6]).astype(np.int64))
    assert np.all(
        lifetime_data.complete.values
        == np.array([10, 9, 11]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(lifetime_data.right_censoring.index == np.array([1, 3, 4, 5]))
    assert np.all(
        lifetime_data.right_censoring.values
        == np.array([11, 10, 12, 13]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.left_truncation.index
        == np.array([2, 3, 4, 5, 6]).astype(np.int64)
    )
    assert np.all(
        lifetime_data.left_truncation.values
        == np.array([3, 5, 3, 1, 9]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.complete.intersection(lifetime_data.left_truncation).values[:, 0]
        == np.array([9, 11]).astype(np.float64)
    )

    assert np.all(
        lifetime_data.complete.intersection(lifetime_data.left_truncation).values[:, 1]
        == np.array([3, 9]).astype(np.float64)
    )

    assert np.all(
        lifetime_data.left_truncation.intersection(lifetime_data.complete).values[:, 0]
        == np.array([3, 9]).astype(np.float64)
    )

    assert np.all(
        lifetime_data.left_truncation.intersection(lifetime_data.complete).values[:, 1]
        == np.array([9, 11]).astype(np.float64)
    )


def test_2d_data(example_2d_data):
    reader = Lifetime2DParser(
        example_2d_data["observed_lifetimes"],
        entry=example_2d_data["entry"],
        departure=example_2d_data["departure"],
    )

    lifetime_data = LifetimeData(
        len(example_2d_data),
        reader.get_complete(),
        reader.get_left_censoring(),
        reader.get_right_censoring(),
        reader.get_interval_censoring(),
        reader.get_left_truncation(),
        reader.get_right_truncation(),
    )

    assert np.all(lifetime_data.left_censoring.index == np.array([1]).astype(np.int64))
    assert np.all(
        lifetime_data.left_censoring.values
        == np.array([4]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(lifetime_data.right_censoring.index == np.array([3]))
    assert np.all(
        lifetime_data.right_censoring.values
        == np.array([7]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.interval_censoring.index
        == np.array([0, 1, 3, 5, 6]).astype(np.int64)
    )
    assert np.all(
        lifetime_data.interval_censoring.values
        == np.array([[1, 2], [0, 4], [7, np.inf], [2, 10], [10, 11]]).astype(np.float64)
    )

    assert np.all(
        lifetime_data.left_truncation.values == np.array([3, 5, 3, 1, 9]).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.left_truncation.intersection(
            lifetime_data.interval_censoring
        ).index
        == np.array([3, 5, 6]).astype(np.int64)
    )

    assert np.all(
        lifetime_data.left_truncation.intersection(
            lifetime_data.interval_censoring
        ).values[:, 1:]
        == np.array([[7, np.inf], [2, 10], [10, 11]]).astype(np.float64)
    )

    assert np.all(
        lifetime_data.right_censoring.intersection(
            lifetime_data.left_truncation
        ).values[:, 0]
        == np.array([7]).astype(np.float64)
    )
    assert np.all(
        lifetime_data.right_censoring.intersection(lifetime_data.left_truncation).index
        == np.array([3]).astype(np.int64)
    )
