import numpy as np
import pytest

from relife2.data import databook


@pytest.fixture
def example_1d_data():
    return {
        "lifetimes": np.array([10, 11, 9, 10, 12, 13, 11]),
        "event": np.array([1, 0, 1, 0, 0, 0, 1]),
        "entry": np.array([0, 0, 3, 5, 3, 1, 9]),
    }


@pytest.fixture
def example_2d_data():
    return {
        "lifetimes": np.array(
            [[1, 2], [0, 4], [5, 5], [7, np.inf], [10, 10], [2, 10], [10, 11]]
        ),
        "entry": np.array([0, 0, 3, 5, 3, 1, 9]),
    }


def test_1d_data(example_1d_data):
    db = databook(
        censored_lifetimes=example_1d_data["lifetimes"],
        right_censored_indicators=example_1d_data["event"] == 0,
        observed_indicators=example_1d_data["event"] == 1,
        entry=example_1d_data["entry"],
    )
    assert (db.observed.index == np.array([0, 2, 6])).all()
    assert (db.observed.values == np.array([10, 9, 11])).all()
    assert (db("observed").index == np.array([0, 2, 6])).all()
    assert (db("observed").values == np.array([10, 9, 11])).all()

    assert (db.right_censored.index == np.array([1, 3, 4, 5])).all()
    assert (db.right_censored.values == np.array([11, 10, 12, 13])).all()
    assert (db("right_censored").index == np.array([1, 3, 4, 5])).all()
    assert (db("right_censored").values == np.array([11, 10, 12, 13])).all()

    assert (db.left_truncated.index == np.array([2, 3, 4, 5, 6])).all()
    assert (db.left_truncated.values == np.array([3, 5, 3, 1, 9])).all()
    assert (db("left_truncated").index == np.array([2, 3, 4, 5, 6])).all()
    assert (db("left_truncated").values == np.array([3, 5, 3, 1, 9])).all()

    assert (db("observed & left_truncated")[0].values == np.array([9, 11])).all()

    assert (db("observed & left_truncated")[1].values == np.array([3, 9])).all()

    assert (
        db("left_truncated & observed & right_truncated")[0].values == np.array([3, 9])
    ).all()

    assert (db("left_truncated & observed")[1].values == np.array([9, 11])).all()


# def test_censored(example_2d_data):
#     data = SurvivalData(**example_2d_data)

#     # left censored
#     assert (data.censored(how="left") == np.array([1])).all()
#     assert (data.censored(how="left", return_values=True) == np.array([4])).all()

#     # right censored
#     assert (data.censored(how="right") == np.array([3])).all()
#     assert (data.censored(how="right", return_values=True) == np.array([7])).all()

#     # interval censored
#     assert (data.censored(how="interval") == np.array([0, 5, 6])).all()
#     assert (
#         data.censored(how="interval", return_values=True)
#         == np.array([[1, 2], [2, 10], [10, 11]])
#     ).all()


# def test_truncated(example_2d_data_with_truncations):
#     data = SurvivalData(**example_2d_data_with_truncations)

#     # left truncations
#     assert (data.truncated(how="left") == np.array([1, 2, 3, 4])).all()
#     assert (
#         data.truncated(how="left", return_values=True) == np.array([2, 3, 10, 3])
#     ).all()
