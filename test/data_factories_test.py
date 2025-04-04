import numpy as np
import pytest

from relife.data import lifetime_data_factory, nhpp_data_factory

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


@pytest.fixture
def nhpp_data_v0():
    return {
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "ages": (11, 13, 21, 25, 27),
    }


@pytest.fixture
def nhpp_data_v1():
    return {
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "ages": (11, 13, 21, 25, 27),
        "assets_ids": ("AB2", "CX13"),
        "first_ages": (10, 12),
        "last_ages": (35, 60),
        "model_args": (np.array([1.2, 5.5]), np.array([37.2, 22.2])),
    }


@pytest.fixture
def nhpp_data_v2():
    return {
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "ages": (11, 13, 21, 25, 27),
        "assets_ids": ("AB2", "CX13"),
        "model_args": (np.array([1.2, 5.5]), np.array([37.2, 22.2])),
    }


@pytest.fixture
def nhpp_data_v3():
    return {
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "ages": (11, 13, 21, 25, 27),
        "assets_ids": ("AB2", "CX13"),
        "first_ages": (10, 12),
        "model_args": (np.array([1.2, 5.5]), np.array([37.2, 22.2])),
    }


@pytest.fixture
def nhpp_data_v4():
    return {
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "ages": (11, 13, 21, 25, 27),
        "assets_ids": ("AB2", "CX13"),
        "last_ages": (35, 60),
        "model_args": (np.array([1.2, 5.5]), np.array([37.2, 22.2])),
    }


def test_1d_data(example_1d_data):
    lifetime_data = lifetime_data_factory(
        example_1d_data["observed_lifetimes"],
        event=example_1d_data["event"],
        entry=example_1d_data["entry"],
    )

    assert np.all(lifetime_data.complete.index == np.array([0, 2, 6]).astype(np.int64))
    assert np.all(
        lifetime_data.complete.values
        == np.array([10, 9, 11]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(lifetime_data.right_censoring.index == np.array([1, 3, 4, 5]))
    assert np.all(
        lifetime_data.right_censoring.values
        == np.array([11, 10, 12, 13]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.left_truncation.index
        == np.array([2, 3, 4, 5, 6]).astype(np.int64)
    )
    assert np.all(
        lifetime_data.left_truncation.values
        == np.array([3, 5, 3, 1, 9]).astype(np.float64).reshape(-1, 1)
    )

    # assert np.all(
    #     lifetime_data.complete.intersection(lifetime_data.left_truncation).values[:, 0]
    #     == np.array([9, 11]).astype(np.float64)
    # )
    #
    # assert np.all(
    #     lifetime_data.complete.intersection(lifetime_data.left_truncation).values[:, 1]
    #     == np.array([3, 9]).astype(np.float64)
    # )
    #
    # assert np.all(
    #     lifetime_data.left_truncation.intersection(lifetime_data.complete).values[:, 0]
    #     == np.array([3, 9]).astype(np.float64)
    # )
    #
    # assert np.all(
    #     lifetime_data.left_truncation.intersection(lifetime_data.complete).values[:, 1]
    #     == np.array([9, 11]).astype(np.float64)
    # )


def test_2d_data(example_2d_data):
    lifetime_data = lifetime_data_factory(
        example_2d_data["observed_lifetimes"],
        entry=example_2d_data["entry"],
        departure=example_2d_data["departure"],
    )

    assert np.all(lifetime_data.left_censoring.index == np.array([1]).astype(np.int64))
    assert np.all(
        lifetime_data.left_censoring.values
        == np.array([4]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(lifetime_data.right_censoring.index == np.array([3]))
    assert np.all(
        lifetime_data.right_censoring.values
        == np.array([7]).astype(np.float64).reshape(-1, 1)
    )

    assert np.all(
        lifetime_data.interval_censoring.index
        == np.array([0, 1, 3, 5, 6]).astype(np.int64)
    )
    assert np.all(
        lifetime_data.interval_censoring.values
        == np.array([[1, 2], [0, 4], [7, np.inf], [2, 10], [10, 11]]).astype(np.float64)
    )

    assert np.all(
        lifetime_data.left_truncation.values == np.array([3, 5, 3, 1, 9]).reshape(-1, 1)
    )

    # assert np.all(
    #     lifetime_data.left_truncation.intersection(
    #         lifetime_data.interval_censoring
    #     ).index
    #     == np.array([3, 5, 6]).astype(np.int64)
    # )
    #
    # assert np.all(
    #     lifetime_data.left_truncation.intersection(
    #         lifetime_data.interval_censoring
    #     ).values[:, 1:]
    #     == np.array([[7, np.inf], [2, 10], [10, 11]]).astype(np.float64)
    # )
    #
    # assert np.all(
    #     lifetime_data.right_censoring.intersection(
    #         lifetime_data.left_truncation
    #     ).values[:, 0]
    #     == np.array([7]).astype(np.float64)
    # )
    # assert np.all(
    #     lifetime_data.right_censoring.intersection(lifetime_data.left_truncation).index
    #     == np.array([3]).astype(np.int64)
    # )


def test_nhhp_data_v0(nhpp_data_v0):
    time, event, entry, model_args = nhpp_data_factory(**nhpp_data_v0)
    assert np.all(time == np.array([11.0, 21.0, 25.0, 13.0, 27.0]))
    assert np.all(event == np.array([True, True, True, True, True]))
    assert np.all(entry == np.array([0.0, 11.0, 21.0, 0.0, 13.0]))
    assert model_args == ()


def test_nhhp_data_v1(nhpp_data_v1):
    time, event, entry, model_args = nhpp_data_factory(**nhpp_data_v1)
    assert np.all(time == np.array([11.0, 21.0, 25.0, 35.0, 13.0, 27.0, 60.0]))
    assert np.all(event == np.array([True, True, True, False, True, True, False]))
    assert np.all(entry == np.array([10.0, 11.0, 21.0, 25.0, 12.0, 13.0, 27.0]))
    assert np.all(model_args[0] == np.array([1.2, 1.2, 1.2, 1.2, 5.5, 5.5, 5.5]))
    assert np.all(model_args[1] == np.array([37.2, 37.2, 37.2, 37.2, 22.2, 22.2, 22.2]))


def test_nhhp_data_v2(nhpp_data_v2):
    time, event, entry, model_args = nhpp_data_factory(**nhpp_data_v2)
    assert np.all(time == np.array([11.0, 21.0, 25.0, 13.0, 27.0]))
    assert np.all(event == np.array([True, True, True, True, True]))
    assert np.all(entry == np.array([0.0, 11.0, 21.0, 0.0, 13.0]))
    assert np.all(model_args[0] == np.array([1.2, 1.2, 1.2, 5.5, 5.5]))
    assert np.all(model_args[1] == np.array([37.2, 37.2, 37.2, 22.2, 22.2]))


def test_nhhp_data_v3(nhpp_data_v3):
    time, event, entry, model_args = nhpp_data_factory(**nhpp_data_v3)
    assert np.all(time == np.array([11.0, 21.0, 25.0, 13.0, 27.0]))
    assert np.all(event == np.array([True, True, True, True, True]))
    assert np.all(entry == np.array([10.0, 11.0, 21.0, 12.0, 13.0]))
    assert np.all(model_args[0] == np.array([1.2, 1.2, 1.2, 5.5, 5.5]))
    assert np.all(model_args[1] == np.array([37.2, 37.2, 37.2, 22.2, 22.2]))


def test_nhhp_data_v4(nhpp_data_v4):
    time, event, entry, model_args = nhpp_data_factory(**nhpp_data_v4)
    assert np.all(time == np.array([11.0, 21.0, 25.0, 35.0, 13.0, 27.0, 60.0]))
    assert np.all(event == np.array([True, True, True, False, True, True, False]))
    assert np.all(entry == np.array([0.0, 11.0, 21.0, 25.0, 0.0, 13.0, 27.0]))
    assert np.all(model_args[0] == np.array([1.2, 1.2, 1.2, 1.2, 5.5, 5.5, 5.5]))
    assert np.all(model_args[1] == np.array([37.2, 37.2, 37.2, 37.2, 22.2, 22.2, 22.2]))
