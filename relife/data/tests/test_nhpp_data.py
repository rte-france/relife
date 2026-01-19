# pyright: basic
import numpy as np
import pytest

from relife.data import NHPPData


@pytest.fixture
def nhpp_data_v0():
    return {
        "ages_at_events": (11, 13, 21, 25, 27),
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
    }


@pytest.fixture
def nhpp_data_v1():
    return {
        "ages_at_events": (11, 13, 21, 25, 27),
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "first_ages": (10, 12),
        "last_ages": (35, 60),
        "model_args": (np.array([1.2, 5.5]), np.array([37.2, 22.2])),
        "assets_ids": ("AB2", "CX13"),
    }


@pytest.fixture
def nhpp_data_v2():
    return {
        "ages_at_events": (11, 13, 21, 25, 27),
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "model_args": (np.array([1.2, 5.5]), np.array([37.2, 22.2])),
        "assets_ids": ("AB2", "CX13"),
    }


@pytest.fixture
def nhpp_data_v3():
    return {
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "assets_ids": ("AB2", "CX13"),
        "first_ages": (10, 12),
        "model_args": (np.array([1.2, 5.5]), np.array([37.2, 22.2])),
        "ages_at_events": (11, 13, 21, 25, 27),
    }


@pytest.fixture
def nhpp_data_v4():
    return {
        "ages_at_events": (11, 13, 21, 25, 27),
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "last_ages": (35, 60),
        "model_args": (np.array([1.2, 5.5]), np.array([37.2, 22.2])),
        "assets_ids": ("AB2", "CX13"),
    }


def test_nhhp_data_v0(nhpp_data_v0):
    time, event, entry, model_args = NHPPData(**nhpp_data_v0).to_lifetime_data()
    assert np.all(time == np.array([11.0, 21.0, 25.0, 13.0, 27.0]))
    assert np.all(event == np.array([True, True, True, True, True]))
    assert np.all(entry == np.array([0.0, 11.0, 21.0, 0.0, 13.0]))
    assert model_args == ()


def test_nhhp_data_v1(nhpp_data_v1):
    time, event, entry, model_args = NHPPData(**nhpp_data_v1).to_lifetime_data()
    assert np.all(time == np.array([11.0, 21.0, 25.0, 35.0, 13.0, 27.0, 60.0]))
    assert np.all(event == np.array([True, True, True, False, True, True, False]))
    assert np.all(entry == np.array([10.0, 11.0, 21.0, 25.0, 12.0, 13.0, 27.0]))
    assert np.all(model_args[0] == np.array([1.2, 1.2, 1.2, 1.2, 5.5, 5.5, 5.5]))
    assert np.all(model_args[1] == np.array([37.2, 37.2, 37.2, 37.2, 22.2, 22.2, 22.2]))


def test_nhhp_data_v2(nhpp_data_v2):
    time, event, entry, model_args = NHPPData(**nhpp_data_v2).to_lifetime_data()
    assert np.all(time == np.array([11.0, 21.0, 25.0, 13.0, 27.0]))
    assert np.all(event == np.array([True, True, True, True, True]))
    assert np.all(entry == np.array([0.0, 11.0, 21.0, 0.0, 13.0]))
    assert np.all(model_args[0] == np.array([1.2, 1.2, 1.2, 5.5, 5.5]))
    assert np.all(model_args[1] == np.array([37.2, 37.2, 37.2, 22.2, 22.2]))


def test_nhhp_data_v3(nhpp_data_v3):
    time, event, entry, model_args = NHPPData(**nhpp_data_v3).to_lifetime_data()
    assert np.all(time == np.array([11.0, 21.0, 25.0, 13.0, 27.0]))
    assert np.all(event == np.array([True, True, True, True, True]))
    assert np.all(entry == np.array([10.0, 11.0, 21.0, 12.0, 13.0]))
    assert np.all(model_args[0] == np.array([1.2, 1.2, 1.2, 5.5, 5.5]))
    assert np.all(model_args[1] == np.array([37.2, 37.2, 37.2, 22.2, 22.2]))


def test_nhhp_data_v4(nhpp_data_v4):
    time, event, entry, model_args = NHPPData(**nhpp_data_v4).to_lifetime_data()
    assert np.all(time == np.array([11.0, 21.0, 25.0, 35.0, 13.0, 27.0, 60.0]))
    assert np.all(event == np.array([True, True, True, False, True, True, False]))
    assert np.all(entry == np.array([0.0, 11.0, 21.0, 25.0, 0.0, 13.0, 27.0]))
    assert np.all(model_args[0] == np.array([1.2, 1.2, 1.2, 1.2, 5.5, 5.5, 5.5]))
    assert np.all(model_args[1] == np.array([37.2, 37.2, 37.2, 37.2, 22.2, 22.2, 22.2]))


#
# DATA_V0 = {
#     "assets_ids": np.array(
#         ["S00", "S00", "S01", "S01", "S02", "S11", "S12", "S12", "S13"], dtype=np.str_
#     ),
#     "timeline": np.array(
#         [
#             5.01409347,
#             8.0,
#             4.0485904,
#             8.0,
#             4.0,
#             8.0,
#             5.07210737,
#             8.0,
#             4.0,
#         ],
#         dtype=np.float64,
#     ),
#     "ages": np.array(
#         [5.01409347, 8.0, 4.0485904, 8.0, 4.0, 8.0, 5.07210737, 8.0, 4.0],
#         dtype=np.float64,
#     ),
#     "events_indicators": np.array(
#         [True, False, True, False, False, False, True, False, False], dtype=np.bool_
#     ),
#     "entries": np.array(
#         [2.0, 5.01409347, 0.0, 4.0485904, 0.0, 2.0, 0.0, 5.07210737, 0.0],
#         dtype=np.float64,
#     ),
# }
#
# @pytest.mark.skip(reason="no way of currently testing this")
# def _test_nhpp_to_failure_data(data):
#
#     timeline = data["timeline"]
#     assets_ids = data["assets_ids"]
#     ages = data["ages"]
#     entries = data["entries"]
#     events_indicators = data["events_indicators"]
#
#     sort_ind = np.lexsort((timeline, assets_ids))
#
#     entries = entries[sort_ind]
#     events_indicators = events_indicators[sort_ind]
#     ages = ages[sort_ind]
#     assets_ids = assets_ids[sort_ind]
#
#     first_ages_index = np.roll(assets_ids, 1) != assets_ids
#     last_ages_index = np.roll(first_ages_index, -1)
#
#     immediatly_replaced = np.logical_and(~events_indicators, first_ages_index)
#
#     prefix = np.full_like(assets_ids[immediatly_replaced], "Z", dtype=np.str_)
#     _assets_ids = np.char.add(prefix, assets_ids[immediatly_replaced])
#     first_ages = entries[immediatly_replaced].copy()
#     last_ages = ages[immediatly_replaced].copy()
#
#     events_assets_ids = assets_ids[events_indicators]
#     events_ages = ages[events_indicators]
#     other_assets_ids = np.unique(events_assets_ids)
#     _assets_ids = np.concatenate((_assets_ids, other_assets_ids))
#     first_ages = np.concatenate(
#         (first_ages, entries[first_ages_index & events_indicators])
#     )
#     last_ages = np.concatenate(
#         (last_ages, ages[last_ages_index & ~immediatly_replaced])
#     )
#
#     return events_assets_ids, events_ages, _assets_ids, first_ages, last_ages
#
# @pytest.mark.skip(reason="no way of currently testing this")
# def test_nhpp_to_failure_data():
#
#     events_assets_ids, events_ages, assets_ids, first_ages, last_ages = (
#         _test_nhpp_to_failure_data(DATA_V0)
#     )
#
#     assert np.all(events_assets_ids == np.array(["S00", "S01", "S12"], dtype=np.str_))
#     assert np.all(
#         events_ages == np.array([5.01409347, 4.0485904, 5.07210737], dtype=np.float64)
#     )
#     assert np.all(
#         assets_ids
#         == np.array(["ZS02", "ZS11", "ZS13", "S00", "S01", "S12"], dtype=np.str_)
#     )
#     assert np.all(
#         first_ages == np.array([0.0, 2.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float64)
#     )
#     assert np.all(
#         last_ages == np.array([4.0, 8.0, 4.0, 8.0, 8.0, 8.0], dtype=np.float64)
#     )
