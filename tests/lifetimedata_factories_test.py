import numpy as np
import pytest

from relife2.data import intersect_lifetimes
from relife2.data.factories import (
    LifetimeDataFactoryFrom1D,
    LifetimeDataFactoryFrom2D,
    array_factory,
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
        event=example_1d_data["event"],
        entry=example_1d_data["entry"],
    )

    observed_lifetimes, truncations = factory()

    assert np.all(observed_lifetimes.complete.ids == np.array([0, 2, 6]))
    assert np.all(
        observed_lifetimes.complete.values == np.array([10, 9, 11]).reshape(-1, 1)
    )

    assert np.all(observed_lifetimes.right_censored.ids == np.array([1, 3, 4, 5]))
    assert np.all(
        observed_lifetimes.right_censored.values
        == np.array([11, 10, 12, 13]).reshape(-1, 1)
    )

    assert np.all(truncations.left.ids == np.array([2, 3, 4, 5, 6]))
    assert np.all(truncations.left.values == np.array([3, 5, 3, 1, 9]).reshape(-1, 1))

    assert np.all(
        intersect_lifetimes(observed_lifetimes.complete, truncations.left)[0].values
        == np.array([9, 11]).reshape(-1, 1)
    )

    assert np.all(
        intersect_lifetimes(observed_lifetimes.complete, truncations.left)[1].values
        == np.array([3, 9]).reshape(-1, 1)
    )

    assert np.all(
        intersect_lifetimes(truncations.left, observed_lifetimes.complete)[0].values
        == np.array([3, 9]).reshape(-1, 1)
    )

    assert np.all(
        intersect_lifetimes(truncations.left, observed_lifetimes.complete)[1].values
        == np.array([9, 11]).reshape(-1, 1)
    )


def test_2d_data(example_2d_data):
    factory = LifetimeDataFactoryFrom2D(
        example_2d_data["observed_lifetimes"],
        entry=example_2d_data["entry"],
        departure=example_2d_data["departure"],
    )

    observed_lifetimes, truncations = factory()

    assert np.all(observed_lifetimes.left_censored.ids == np.array([1]))
    assert np.all(
        observed_lifetimes.left_censored.values == np.array([4]).reshape(-1, 1)
    )

    assert np.all(observed_lifetimes.right_censored.ids == np.array([3]))
    assert np.all(
        observed_lifetimes.right_censored.values == np.array([7]).reshape(-1, 1)
    )

    assert np.all(observed_lifetimes.interval_censored.ids == np.array([0, 1, 3, 5, 6]))
    assert np.all(
        observed_lifetimes.interval_censored.values
        == np.array([[1, 2], [0, 4], [7, np.inf], [2, 10], [10, 11]])
    )

    assert np.all(truncations.left.values == np.array([3, 5, 3, 1, 9]).reshape(-1, 1))

    assert np.all(
        intersect_lifetimes(truncations.left, observed_lifetimes.interval_censored)[
            1
        ].ids
        == np.array([3, 5, 6])
    )

    assert np.all(
        intersect_lifetimes(truncations.left, observed_lifetimes.interval_censored)[
            1
        ].values
        == np.array([[7, np.inf], [2, 10], [10, 11]])
    )

    assert np.all(
        intersect_lifetimes(observed_lifetimes.right_censored, truncations.left)[
            0
        ].values
        == np.array([7]).reshape(-1, 1)
    )
    assert np.all(
        intersect_lifetimes(observed_lifetimes.right_censored, truncations.left)[0].ids
        == np.array([3])
    )
