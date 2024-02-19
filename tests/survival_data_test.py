import numpy as np
import pytest

import relife2.data as rd


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
    data = rd.survdata(
        censored_lifetimes=example_1d_data["lifetimes"],
        right_censored_indicators=example_1d_data["event"] == 0,
        observed_indicators=example_1d_data["event"] == 1,
        entry=example_1d_data["entry"],
    )
    assert (data.observed.index == np.array([0, 2, 6])).all()
    assert (data.observed.values == np.array([10, 9, 11])).all()
    assert (data.index("observed") == np.array([0, 2, 6])).all()
    assert (data.values("observed") == np.array([10, 9, 11])).all()

    assert (data.right_censored.index == np.array([1, 3, 4, 5])).all()
    assert (data.right_censored.values == np.array([11, 10, 12, 13])).all()
    assert (data.index("right_censored") == np.array([1, 3, 4, 5])).all()
    assert (data.values("right_censored") == np.array([11, 10, 12, 13])).all()

    assert (data.left_truncated.index == np.array([2, 3, 4, 5, 6])).all()
    assert (data.left_truncated.values == np.array([3, 5, 3, 1, 9])).all()
    assert (data.index("left_truncated") == np.array([2, 3, 4, 5, 6])).all()
    assert (data.values("left_truncated") == np.array([3, 5, 3, 1, 9])).all()

    assert (
        data.intersection_values("observed", "left_truncated")[0] == np.array([9, 11])
    ).all()

    assert (
        data.intersection_values("observed", "left_truncated")[1] == np.array([3, 9])
    ).all()

    assert (
        data.intersection_values("left_truncated", "observed")[0] == np.array([3, 9])
    ).all()

    assert (
        data.intersection_values("left_truncated", "observed")[1] == np.array([9, 11])
    ).all()


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
