import numpy as np
import pytest

from relife2.survival.data import databook


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
    db = databook(
        observed_lifetimes=example_1d_data["observed_lifetimes"],
        right_censored_indicators=example_1d_data["event"] == 0,
        complete_indicators=example_1d_data["event"] == 1,
        entry=example_1d_data["entry"],
    )
    assert (db.complete.index == np.array([0, 2, 6])).all()
    assert (db.complete.values == np.array([10, 9, 11])).all()
    assert (db("complete").index == np.array([0, 2, 6])).all()
    assert (db("complete").values == np.array([10, 9, 11])).all()

    assert (db.right_censored.index == np.array([1, 3, 4, 5])).all()
    assert (db.right_censored.values == np.array([11, 10, 12, 13])).all()
    assert (db("right_censored").index == np.array([1, 3, 4, 5])).all()
    assert (db("right_censored").values == np.array([11, 10, 12, 13])).all()

    assert (db.left_truncated.index == np.array([2, 3, 4, 5, 6])).all()
    assert (db.left_truncated.values == np.array([3, 5, 3, 1, 9])).all()
    assert (db("left_truncated").index == np.array([2, 3, 4, 5, 6])).all()
    assert (db("left_truncated").values == np.array([3, 5, 3, 1, 9])).all()

    assert (
        db("complete & left_truncated")[0].values == np.array([9, 11])
    ).all()

    assert (
        db("complete & left_truncated")[1].values == np.array([3, 9])
    ).all()

    assert (
        db("left_truncated & complete")[0].values == np.array([3, 9])
    ).all()

    assert (
        db("left_truncated & complete")[1].values == np.array([9, 11])
    ).all()


def test_2d_data(example_2d_data):
    db = databook(observed_lifetimes=example_2d_data["observed_lifetimes"])

    # left censored
    assert (db("left_censored").index == np.array([1])).all()
    assert (db("left_censored").values == np.array([4])).all()

    # right censored
    assert (db("right_censored").index == np.array([3])).all()
    assert (db("right_censored").values == np.array([7])).all()

    # interval censored
    assert (db("interval_censored").index == np.array([0, 5, 6])).all()
    assert (
        db("interval_censored").values == np.array([[1, 2], [2, 10], [10, 11]])
    ).all()


def test_2d_data_with_left_truncations(example_2d_data):
    db = databook(
        observed_lifetimes=example_2d_data["observed_lifetimes"],
        entry=example_2d_data["entry"],
    )

    # left truncations
    assert (db("left_truncated").index == np.array([2, 3, 4, 5, 6])).all()
    assert (db("left_truncated").values == np.array([3, 5, 3, 1, 9])).all()

    assert (
        db("left_truncated & interval_censored")[0].values == np.array([1, 9])
    ).all()
    assert (
        db("left_truncated & interval_censored")[0].index == np.array([5, 6])
    ).all()
    assert (
        db("left_truncated & interval_censored")[1].values
        == np.array([[2, 10], [10, 11]])
    ).all()
    assert (
        db("left_truncated & interval_censored")[1].index == np.array([5, 6])
    ).all()

    assert (
        db("right_censored & left_truncated")[0].values == np.array([7])
    ).all()
    assert (
        db("right_censored & left_truncated")[1].values == np.array([5])
    ).all()


def test_2d_data_with_all_truncations(example_2d_data):
    db = databook(
        observed_lifetimes=example_2d_data["observed_lifetimes"],
        entry=example_2d_data["entry"],
        departure=example_2d_data["departure"],
    )

    # left truncations
    assert (db("left_truncated").index == np.array([4, 6])).all()
    assert (db("left_truncated").values == np.array([3, 9])).all()

    assert (
        db("left_truncated & interval_censored")[0].values == np.array([9])
    ).all()
    assert (
        db("left_truncated & interval_censored")[0].index == np.array([6])
    ).all()
    assert (
        db("left_truncated & interval_censored")[1].values
        == np.array([[10, 11]])
    ).all()
    assert (
        db("left_truncated & interval_censored")[1].index == np.array([6])
    ).all()

    assert (
        db("right_censored & left_truncated")[0].values == np.array([])
    ).all()
    assert (
        db("right_censored & left_truncated")[1].values == np.array([])
    ).all()
