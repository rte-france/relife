import numpy as np
from relife.utils.data import RenewalData


def test_2_samples_1_asset():
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

    event_times = np.array([40., 36., 66., 77., 87., 112., 123., 132.])
    lifetimes = np.array([40., 36., 26., 41., 21., 35., 36., 20.])
    assets_index = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    samples_index = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    order = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)

    sample_data = RenewalData(
        samples_index,
        assets_index,
        order,
        event_times,
        lifetimes,
        np.ones_like(lifetimes, dtype=np.bool_),  # events
        (),  # model args
        False,  # with model1
    )

    t0 = 90.
    tf = 120.

    time, event, entry, departure, model_args = sample_data.to_fit(t0, tf)

    assert np.all(time == np.array([35., 33., 8.]))
    assert np.all(event == np.array([True, False, False]))
    assert np.all(entry == np.array([13., 0., 0.]))


def test_2_samples_2_assets():
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

    event_times = np.array([40., 36., 38., 35., 66., 77., 71., 74., 87., 112., 92., 89., 123., 132., 119., 126.])
    lifetimes = np.array([40., 36., 38., 35., 26., 41., 33., 39., 21., 35., 21., 15., 36., 20., 27., 37.])
    assets_index = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    samples_index = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    order = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int64)

    sample_data = RenewalData(
        samples_index,
        assets_index,
        order,
        event_times,
        lifetimes,
        np.ones_like(lifetimes, dtype=np.bool_),  # events
        (),  # model args
        False,  # with model1
    )

    t0 = 90.
    tf = 120.

    time, event, entry, departure, model_args = sample_data.to_fit(t0, tf)

    assert np.all(time == np.array([35., 21., 27., 33., 8., 31.]))
    assert np.all(event == np.array([True, True, True, False, False, False]))
    assert np.all(entry == np.array([13., 19., 0., 0., 0., 0.]))