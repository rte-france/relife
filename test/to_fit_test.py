import numpy as np
import pytest

from relife.core.descriptors import ShapedArgs
from relife.data import RenewalData


@pytest.fixture
def samples2_assets1():
    """
    event_times
    (sample, asset) values...
    (0, 0) 40 66 87 123
    (1, 0) 36 77 112 132
    lifetimes
    (sample, asset) values...
    (0, 0) 40 26 21 36
    (1, 0) 36 41 35 20
    """
    event_times = np.array([40.0, 36.0, 66.0, 77.0, 87.0, 112.0, 123.0, 132.0])
    lifetimes = np.array([40.0, 36.0, 26.0, 41.0, 21.0, 35.0, 36.0, 20.0])
    assets_index = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    samples_index = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

    sample_data = RenewalData(
        samples_index,
        assets_index,
        event_times,
        lifetimes,
        np.ones_like(lifetimes, dtype=np.bool_),  # events
        (),  # core args
        False,  # with model1
    )
    return sample_data


# def args_in_2d(model_args)
#     args_2d = [np.atleast_2d(arg) for arg in model_args]
#     nb_assets =
#     if nb_assets > 1 and bool(model_args):
#         for i, arg in enumerate(args_2d):
#             if arg.shape[0] == 1:
#                 args_2d[i] = np.tile(arg, (self.nb_assets, 1))
#
#
#
#


@pytest.fixture(
    params=[
        (np.array([[1], [1]]),),
        (np.array([[1, 2, 3], [1, 2, 3]]),),
        (np.array([[1, 2, 3], [4, 5, 6]]),),
        (
            np.array([[1], [1]]),
            np.array([[1, 2, 3], [1, 2, 3]]),
            np.array([[1, 2, 3], [4, 5, 6]]),
        ),
    ]
)
def samples2_assets2(request):
    """
    event_times
    (sample, asset) values...
    (0, 0) 40 66 87 123
    (1, 0) 36 77 112 132
    (0, 1) 38 71 92 119
    (1, 1) 35 74 89 126
    lifetimes
    (sample, asset) values...
    (0, 0) 40 26 21 36
    (1, 0) 36 41 35 20
    (0, 1) 38 33 21 27
    (1, 1) 35 39 15 37
    """
    event_times = np.array(
        [
            40.0,
            36.0,
            38.0,
            35.0,
            66.0,
            77.0,
            71.0,
            74.0,
            87.0,
            112.0,
            92.0,
            89.0,
            123.0,
            132.0,
            119.0,
            126.0,
        ]
    )
    lifetimes = np.array(
        [
            40.0,
            36.0,
            38.0,
            35.0,
            26.0,
            41.0,
            33.0,
            39.0,
            21.0,
            35.0,
            21.0,
            15.0,
            36.0,
            20.0,
            27.0,
            37.0,
        ]
    )
    assets_index = np.array(
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64
    )
    samples_index = np.array(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64
    )

    model_args = request.param

    sample_data = RenewalData(
        samples_index,
        assets_index,
        event_times,
        lifetimes,
        np.ones_like(lifetimes, dtype=np.bool_),  # events
        model_args,
        False,  # with model1
    )
    return sample_data


def test_2_samples_1_asset(samples2_assets1):
    t0 = 90.0
    tf = 120.0

    time, event, entry, departure, model_args = samples2_assets1.to_fit(t0, tf)

    assert np.all(time == np.array([35.0, 33.0, 8.0]))
    assert np.all(event == np.array([True, False, False]))
    assert np.all(entry == np.array([13.0, 0.0, 0.0]))


def test_2_samples_2_assets(samples2_assets2):
    t0 = 90.0
    tf = 120.0

    time, event, entry, departure, model_args = samples2_assets2.to_fit(t0, tf)

    assert np.all(time == np.array([35.0, 21.0, 27.0, 33.0, 8.0, 31.0]))
    assert np.all(event == np.array([True, True, True, False, False, False]))
    assert np.all(entry == np.array([13.0, 19.0, 0.0, 0.0, 0.0, 0.0]))
    assert model_args[0].shape[0] == len(time)
