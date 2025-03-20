import numpy as np
import pytest


DATA_V0 = {
    "assets_ids": np.array(['S00', 'S00', 'S01', 'S01', 'S02', 'S11', 'S12', 'S12', 'S13'], dtype=np.str_),
    "timeline" : np.array([5.01409347, 8., 4.0485904, 8., 4., 8., 5.07210737, 8., 4.,], dtype=np.float64),
    "ages": np.array([5.01409347, 8., 4.0485904, 8., 4., 8., 5.07210737, 8., 4.], dtype=np.float64),
    "events_indicators" : np.array([True, False, True, False, False, False, True, False, False], dtype=np.bool_),
    "entries": np.array([2., 5.01409347, 0., 4.0485904,  0.,         2., 0., 5.07210737, 0.        ], dtype=np.float64),
}


def _test_nhpp_to_failure_data(
    data
):

    timeline = data["timeline"]
    assets_ids = data["assets_ids"]
    ages = data["ages"]
    entries = data["entries"]
    events_indicators = data["events_indicators"]


    sort_ind = np.lexsort((timeline, assets_ids))

    entries = entries[sort_ind]
    events_indicators = events_indicators[sort_ind]
    ages = ages[sort_ind]
    assets_ids = assets_ids[sort_ind]

    first_ages_index = np.roll(assets_ids, 1) != assets_ids
    last_ages_index = np.roll(first_ages_index, -1)

    immediatly_replaced = np.logical_and(~events_indicators, first_ages_index)

    prefix = np.full_like(assets_ids[immediatly_replaced], "Z", dtype=np.str_)
    _assets_ids = np.char.add(prefix, assets_ids[immediatly_replaced])
    first_ages = entries[immediatly_replaced].copy()
    last_ages = ages[immediatly_replaced].copy()

    events_assets_ids = assets_ids[events_indicators]
    events_ages = ages[events_indicators]
    other_assets_ids = np.unique(events_assets_ids)
    _assets_ids = np.concatenate((_assets_ids, other_assets_ids))
    first_ages = np.concatenate((first_ages, entries[first_ages_index & events_indicators]))
    last_ages = np.concatenate((last_ages, ages[last_ages_index & ~immediatly_replaced]))

    return events_assets_ids, events_ages, _assets_ids, first_ages, last_ages


def test_nhpp_to_failure_data():

    events_assets_ids, events_ages, assets_ids, first_ages, last_ages = _test_nhpp_to_failure_data(DATA_V0)

    assert np.all(events_assets_ids == np.array(["S00", "S01", "S12"], dtype=np.str_))
    assert np.all(events_ages == np.array([5.01409347, 4.0485904, 5.07210737], dtype=np.float64))
    assert np.all(assets_ids == np.array(["ZS02", "ZS11", "ZS13", "S00", "S01", "S12"], dtype=np.str_))
    assert np.all(first_ages == np.array([0., 2., 0. ,2. ,0. ,0.], dtype=np.float64))
    assert np.all(last_ages == np.array([4., 8., 4., 8., 8., 8.], dtype=np.float64))