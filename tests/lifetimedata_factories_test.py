import numpy as np
import pytest

from relife2.data import (
    LifetimeDataReaderFrom1D,
    LifetimeDataReaderFrom2D,
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
    reader = LifetimeDataReaderFrom1D(
        example_1d_data["observed_lifetimes"],
        event=example_1d_data["event"],
        entry=example_1d_data["entry"],
    )

    lifetime_data = LifetimeData(
        reader.get_complete(),
        reader.get_left_censorships(),
        reader.get_right_censorships(),
        reader.get_interval_censorships(),
        reader.get_left_truncations(),
        reader.get_right_truncations(),
    )

    assert np.all(lifetime_data.complete.index == np.array([0, 2, 6]).astype(np.int64))
    assert np.all(
        lifetime_data.complete.values
        == np.array([10, 9, 11]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(lifetime_data.right_censored.index == np.array([1, 3, 4, 5]))
    assert np.all(
        lifetime_data.right_censored.values
        == np.array([11, 10, 12, 13]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.left_truncated.index == np.array([2, 3, 4, 5, 6]).astype(np.int64)
    )
    assert np.all(
        lifetime_data.left_truncated.values
        == np.array([3, 5, 3, 1, 9]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.complete.intersection(lifetime_data.left_truncated)[0].values
        == np.array([9, 11]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.complete.intersection(lifetime_data.left_truncated)[1].values
        == np.array([3, 9]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.left_truncated.intersection(lifetime_data.complete)[0].values
        == np.array([3, 9]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.left_truncated.intersection(lifetime_data.complete)[1].values
        == np.array([9, 11]).astype(np.float64).reshape(-1, 1)
    )


def test_2d_data(example_2d_data):
    reader = LifetimeDataReaderFrom2D(
        example_2d_data["observed_lifetimes"],
        entry=example_2d_data["entry"],
        departure=example_2d_data["departure"],
    )

    lifetime_data = LifetimeData(
        reader.get_complete(),
        reader.get_left_censorships(),
        reader.get_right_censorships(),
        reader.get_interval_censorships(),
        reader.get_left_truncations(),
        reader.get_right_truncations(),
    )

    assert np.all(lifetime_data.left_censored.index == np.array([1]).astype(np.int64))
    assert np.all(
        lifetime_data.left_censored.values
        == np.array([4]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(lifetime_data.right_censored.index == np.array([3]))
    assert np.all(
        lifetime_data.right_censored.values
        == np.array([7]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.interval_censored.index
        == np.array([0, 1, 3, 5, 6]).astype(np.int64)
    )
    assert np.all(
        lifetime_data.interval_censored.values
        == np.array([[1, 2], [0, 4], [7, np.inf], [2, 10], [10, 11]]).astype(np.float64)
    )

    assert np.all(
        lifetime_data.left_truncated.values == np.array([3, 5, 3, 1, 9]).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.left_truncated.intersection(lifetime_data.interval_censored)[
            1
        ].index
        == np.array([3, 5, 6]).astype(np.int64)
    )

    assert np.all(
        lifetime_data.left_truncated.intersection(lifetime_data.interval_censored)[
            1
        ].values
        == np.array([[7, np.inf], [2, 10], [10, 11]]).astype(np.float64)
    )

    assert np.all(
        lifetime_data.right_censored.intersection(lifetime_data.left_truncated)[
            0
        ].values
        == np.array([7]).astype(np.float64).reshape(-1, 1)
    )
    assert np.all(
        lifetime_data.right_censored.intersection(lifetime_data.left_truncated)[0].index
        == np.array([3]).astype(np.int64)
    )
