import numpy as np
import pytest

from relife2.survival.data import (
    LifetimeDataFactoryFrom1D,
    LifetimeDataFactoryFrom2D,
    array_factory,
    intersect_lifetimes,
)


@pytest.fixture
def example_1d_data():
    return {
        "observed_lifetimes": array_factory(np.array([10, 11, 9, 10, 12, 13, 11])),
        "event": array_factory(np.array([1, 0, 1, 0, 0, 0, 1])).astype(np.bool_),
        "entry": array_factory(np.array([0, 0, 3, 5, 3, 1, 9])),
    }


@pytest.fixture
def example_2d_data():
    return {
        "observed_lifetimes": array_factory(
            np.array([[1, 2], [0, 4], [5, 5], [7, np.inf], [10, 10], [2, 10], [10, 11]])
        ),
        "entry": array_factory(np.array([0, 0, 3, 5, 3, 1, 9])),
        "departure": array_factory(np.array([4, np.inf, 7, 10, np.inf, 12, np.inf])),
    }


def test_1d_data(example_1d_data):
    factory = LifetimeDataFactoryFrom1D(
        example_1d_data["observed_lifetimes"],
        entry=example_1d_data["entry"],
        rc_indicators=~example_1d_data["event"],
    )

    observed_lifetimes, truncations = factory()

    assert np.all(observed_lifetimes.complete.unit_ids == np.array([0, 2, 6]))
    assert np.all(
        observed_lifetimes.complete.values == np.array([10, 9, 11]).reshape(-1, 1)
    )

    assert np.all(observed_lifetimes.right_censored.unit_ids == np.array([1, 3, 4, 5]))
    assert np.all(
        observed_lifetimes.right_censored.values
        == np.array([11, 10, 12, 13]).reshape(-1, 1)
    )

    assert np.all(truncations.left.unit_ids == np.array([2, 3, 4, 5, 6]))
    assert np.all(truncations.left.values == np.array([3, 5, 3, 1, 9]).reshape(-1, 1))

    assert np.all(
        intersect_lifetimes(observed_lifetimes.complete, truncations.left).values[
            :, [0]
        ]
        == np.array([9, 11]).reshape(-1, 1)
    )

    assert np.all(
        intersect_lifetimes(observed_lifetimes.complete, truncations.left).values[
            :, [1]
        ]
        == np.array([3, 9]).reshape(-1, 1)
    )

    assert np.all(
        intersect_lifetimes(truncations.left, observed_lifetimes.complete).values[
            :, [0]
        ]
        == np.array([3, 9]).reshape(-1, 1)
    )

    assert np.all(
        intersect_lifetimes(truncations.left, observed_lifetimes.complete).values[
            :, [1]
        ]
        == np.array([9, 11]).reshape(-1, 1)
    )


def test_2d_data(example_2d_data):
    factory = LifetimeDataFactoryFrom2D(
        example_2d_data["observed_lifetimes"],
        entry=example_2d_data["entry"],
        departure=example_2d_data["departure"],
    )

    observed_lifetimes, truncations = factory()

    assert np.all(observed_lifetimes.left_censored.unit_ids == np.array([1]))
    assert np.all(
        observed_lifetimes.left_censored.values == np.array([4]).reshape(-1, 1)
    )

    assert np.all(observed_lifetimes.right_censored.unit_ids == np.array([3]))
    assert np.all(
        observed_lifetimes.right_censored.values == np.array([7]).reshape(-1, 1)
    )

    assert np.all(observed_lifetimes.interval_censored.unit_ids == np.array([0, 5, 6]))
    assert np.all(
        observed_lifetimes.interval_censored.values
        == np.array([[1, 2], [2, 10], [10, 11]])
    )

    assert np.all(truncations.left.unit_ids == np.array([2, 3, 4, 5, 6]))
    assert np.all(truncations.left.values == np.array([3, 5, 3, 1, 9]).reshape(-1, 1))

    assert np.all(
        intersect_lifetimes(
            truncations.left, observed_lifetimes.interval_censored
        ).unit_ids
        == np.array([5, 6])
    )

    assert np.all(
        intersect_lifetimes(
            truncations.left, observed_lifetimes.interval_censored
        ).values[:, 1:]
        == np.array([[2, 10], [10, 11]])
    )

    assert np.all(
        intersect_lifetimes(observed_lifetimes.right_censored, truncations.left).values[
            :, [0]
        ]
        == np.array([7]).reshape(-1, 1)
    )
    assert np.all(
        intersect_lifetimes(
            observed_lifetimes.right_censored, truncations.left
        ).unit_ids
        == np.array([3])
    )
