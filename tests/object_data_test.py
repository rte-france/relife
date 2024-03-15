import numpy as np
import pytest

from relife2.survival.data.factory import (
    complete_factory,
    interval_censored_factory,
    interval_truncated_factory,
    left_censored_factory,
    left_truncated_factory,
    right_censored_factory,
    right_truncated_factory,
)
from relife2.survival.data.object import (
    CensoredFromIndicators,
    CompleteObservations,
    CompleteObservationsFromIndicators,
    IntervalCensored,
    IntervalTruncated,
    LeftCensored,
    LeftTruncated,
    RightCensored,
)


@pytest.fixture
def example_1d_data():
    return {
        "observed_lifetimes": np.array([10, 11, 9, 10, 12, 13, 11]),
        "event": np.array([1, 0, 1, 0, 0, 0, 1]),
        "entry": np.array([0, 0, 3, 5, 3, 1, 9]),
        "departure": np.array([15, 16, 13, 0, 14, 0, 0]),
    }


@pytest.fixture
def example_2d_data():
    return {
        "observed_lifetimes": np.array(
            [[1, 2], [0, 4], [5, 5], [7, np.inf], [10, 10], [2, 10], [10, 11]]
        ),
        "entry": np.array([0, 0, 3, 5, 3, 1, 9]),
    }


def test_1d_data(example_1d_data):
    complete = CompleteObservationsFromIndicators(
        example_1d_data["observed_lifetimes"], example_1d_data["event"] == 1
    )
    right_censored = CensoredFromIndicators(
        example_1d_data["observed_lifetimes"], example_1d_data["event"] == 0
    )
    left_truncated = LeftTruncated(
        example_1d_data["entry"], np.zeros_like(example_1d_data["entry"])
    )

    interval_truncated = IntervalTruncated(
        example_1d_data["entry"], example_1d_data["departure"]
    )

    assert (complete.index == np.array([0, 2, 6])).all()
    assert (complete.values == np.array([10, 9, 11])).all()

    assert (right_censored.index == np.array([1, 3, 4, 5])).all()
    assert (right_censored.values == np.array([11, 10, 12, 13])).all()

    assert (left_truncated.index == np.array([2, 3, 4, 5, 6])).all()
    assert (left_truncated.values == np.array([3, 5, 3, 1, 9])).all()

    assert (interval_truncated.index == np.array([2, 4])).all()
    assert (interval_truncated.values == np.array([[3, 13], [3, 14]])).all()


def test_1d_data_factory(example_1d_data):
    complete = complete_factory(
        example_1d_data["observed_lifetimes"], example_1d_data["event"] == 1
    )
    left_censored = left_censored_factory(
        example_1d_data["observed_lifetimes"]
    )
    right_censored = right_censored_factory(
        example_1d_data["observed_lifetimes"], example_1d_data["event"] == 0
    )
    interval_censored = interval_censored_factory(
        example_1d_data["observed_lifetimes"]
    )
    left_truncated = left_truncated_factory(example_1d_data["entry"])
    right_truncated = right_truncated_factory()
    interval_truncated = interval_truncated_factory(
        left_truncation_values=example_1d_data["entry"]
    )

    assert (complete.index == np.array([0, 2, 6])).all()
    assert (complete.values == np.array([10, 9, 11])).all()

    assert (right_censored.index == np.array([1, 3, 4, 5])).all()
    assert (right_censored.values == np.array([11, 10, 12, 13])).all()

    assert (left_truncated.index == np.array([2, 3, 4, 5, 6])).all()
    assert (left_truncated.values == np.array([3, 5, 3, 1, 9])).all()

    assert left_censored.index.size == 0
    assert left_censored.values.size == 0

    assert interval_censored.index.size == 0
    assert interval_censored.values.size == 0

    assert right_truncated.index.size == 0
    assert right_truncated.values.size == 0

    assert interval_truncated.index.size == 0
    assert (
        interval_truncated.values.size == 0
        and interval_truncated.values.shape
        == (
            0,
            2,
        )
    )


def test_2d_data(example_2d_data):
    complete = CompleteObservations(example_2d_data["observed_lifetimes"])
    left_censored = LeftCensored(example_2d_data["observed_lifetimes"])
    right_censored = RightCensored(example_2d_data["observed_lifetimes"])
    interval_censored = IntervalCensored(example_2d_data["observed_lifetimes"])
    left_truncated = LeftTruncated(
        example_2d_data["entry"], np.zeros_like(example_2d_data["entry"])
    )

    assert (complete.index == np.array([2, 4])).all()
    assert (complete.values == np.array([5, 10])).all()

    assert (left_censored.index == np.array([1])).all()
    assert (left_censored.values == np.array([4])).all()

    assert (right_censored.index == np.array([3])).all()
    assert (right_censored.values == np.array([7])).all()

    assert (interval_censored.index == np.array([0, 5, 6])).all()
    assert (
        interval_censored.values == np.array([[1, 2], [2, 10], [10, 11]])
    ).all()

    assert (left_truncated.index == np.array([2, 3, 4, 5, 6])).all()
    assert (left_truncated.values == np.array([3, 5, 3, 1, 9])).all()


def test_2d_data_factory(example_2d_data):
    complete = complete_factory(example_2d_data["observed_lifetimes"])
    left_censored = left_censored_factory(
        example_2d_data["observed_lifetimes"]
    )
    right_censored = right_censored_factory(
        example_2d_data["observed_lifetimes"]
    )
    interval_censored = interval_censored_factory(
        example_2d_data["observed_lifetimes"]
    )
    left_truncated = left_truncated_factory(example_2d_data["entry"])
    right_truncated = right_truncated_factory()
    interval_truncated = interval_truncated_factory(
        left_truncation_values=example_2d_data["entry"]
    )

    assert (complete.index == np.array([2, 4])).all()
    assert (complete.values == np.array([5, 10])).all()

    assert (left_censored.index == np.array([1])).all()
    assert (left_censored.values == np.array([4])).all()

    assert (right_censored.index == np.array([3])).all()
    assert (right_censored.values == np.array([7])).all()

    assert (interval_censored.index == np.array([0, 5, 6])).all()
    assert (
        interval_censored.values == np.array([[1, 2], [2, 10], [10, 11]])
    ).all()

    assert (left_truncated.index == np.array([2, 3, 4, 5, 6])).all()
    assert (left_truncated.values == np.array([3, 5, 3, 1, 9])).all()

    assert right_truncated.index.size == 0
    assert right_truncated.values.size == 0

    assert interval_truncated.index.size == 0
    assert (
        interval_truncated.values.size == 0
        and interval_truncated.values.shape
        == (
            0,
            2,
        )
    )
