import pytest
import numpy as np


def lifetime_input_1d():
    time = np.array([10, 11, 9, 10, 12, 13, 11], dtype=np.float64)
    event = np.array([1, 0, 1, 0, 0, 0, 1], dtype=np.bool_)
    entry = np.array([0, 0, 3, 5, 3, 1, 9], dtype=np.float64)
    return {"time": time, "event": event, "entry": entry}


def lifetime_input_2d():
    time = np.array([[1, 2], [0, 4], [5, 5], [7, np.inf], [10, 10], [2, 10], [10, 11]], dtype=np.float64)
    entry = np.array([0, 0, 3, 5, 3, 1, 9], dtype=np.float64)
    departure = np.array([4, np.inf, 7, 10, np.inf, 12, np.inf], dtype=np.float64)
    return {"time": time, "entry": entry, "departure": departure}


@pytest.fixture(params=[lifetime_input_1d(), lifetime_input_2d()], ids=["1D_time", "2D_time"])
def lifetime_input(request):
    return request.param
