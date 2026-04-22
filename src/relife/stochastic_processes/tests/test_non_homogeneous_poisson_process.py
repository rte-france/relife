# pyright: basic
import numpy as np
import pytest

from relife.lifetime_models._conditional_models import AgeReplacementModel
from relife.stochastic_processes import NonHomogeneousPoissonProcess
from relife.stochastic_processes._non_homogeneous_poisson_process import NHPPData
from relife.stochastic_processes._sample._iterables import (
    NonHomogeneousPoissonProcessIterable,
)
from relife.utils import get_nb_assets

from .utils import select_from_struct


@pytest.fixture
def nhpp_data_v0():
    return {
        "ages_at_events": np.array([11, 13, 21, 25, 27], dtype=np.float64),
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
    }


@pytest.fixture
def nhpp_data_v1():
    return {
        "ages_at_events": np.array([11, 13, 21, 25, 27], dtype=np.float64),
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "first_ages": np.array([10, 12], dtype=np.float64),
        "last_ages": np.array([35, 60], dtype=np.float64),
        "model_args": (np.array([1.2, 5.5]), np.array([37.2, 22.2])),
        "assets_ids": ("AB2", "CX13"),
    }


@pytest.fixture
def nhpp_data_v2():
    return {
        "ages_at_events": np.array([11, 13, 21, 25, 27], dtype=np.float64),
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "model_args": (np.array([1.2, 5.5]), np.array([37.2, 22.2])),
        "assets_ids": ("AB2", "CX13"),
    }


@pytest.fixture
def nhpp_data_v3():
    return {
        "ages_at_events": np.array([11, 13, 21, 25, 27], dtype=np.float64),
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "assets_ids": ("AB2", "CX13"),
        "first_ages": np.array([10, 12], dtype=np.float64),
        "model_args": (np.array([1.2, 5.5]), np.array([37.2, 22.2])),
    }


@pytest.fixture
def nhpp_data_v4():
    return {
        "ages_at_events": np.array([11, 13, 21, 25, 27], dtype=np.float64),
        "events_assets_ids": ("AB2", "CX13", "AB2", "AB2", "CX13"),
        "last_ages": np.array([35, 60], dtype=np.float64),
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


def test_basic_sampling(distribution):
    nhpp = NonHomogeneousPoissonProcess(distribution)
    nb_samples = 10
    t0 = 0.0
    tf = distribution.ppf(0.95)

    iterable = NonHomogeneousPoissonProcessIterable(nhpp, nb_samples, (t0, tf), seed=21)
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # Check NHPP property for each sample
    for i in range(nb_samples):
        select_sample = select_from_struct(struct_array, sample_id=i)
        np.testing.assert_equal(select_sample["time"][:-1], select_sample["entry"][1:])


def test_age_replacement_sampling(distribution, ar):
    nhpp = NonHomogeneousPoissonProcess(distribution)

    trial_model = AgeReplacementModel(distribution).freeze(ar)
    nb_assets = get_nb_assets(*trial_model.args)
    ar_reshaped = trial_model.args[0]

    nb_samples = 10
    t0 = distribution.ppf(0.3)
    tf = 3 * distribution.ppf(0.95)

    iterable = NonHomogeneousPoissonProcessIterable(
        nhpp, nb_samples, (t0, tf), ar=ar, seed=21
    )
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # Check that all times are less than ar for each asset
    for i in range(nb_assets):
        ar_asset = np.atleast_1d(ar_reshaped)[i]
        select_asset = select_from_struct(struct_array, asset_id=i)
        assert (select_asset["time"] <= ar_asset + 1e-5).all()


def test_regression_sampling(frozen_regression):
    nhpp = NonHomogeneousPoissonProcess(frozen_regression)
    nb_assets = frozen_regression.args[0].shape[0]
    t0 = 0.0
    tf = frozen_regression.ppf(0.95).min()
    nb_samples = 10

    iterable = NonHomogeneousPoissonProcessIterable(nhpp, nb_samples, (t0, tf), seed=21)
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # Check NHPP proprerty for each sample and each asset
    for i in range(nb_assets):
        for j in range(nb_samples):
            select_sample = select_from_struct(struct_array, asset_id=i, sample_id=j)
            np.testing.assert_equal(
                select_sample["time"][:-1], select_sample["entry"][1:]
            )


def test_age_replacement_regression_sampling(frozen_regression, ar):
    nhpp = NonHomogeneousPoissonProcess(frozen_regression)

    trial_model = AgeReplacementModel(frozen_regression).freeze(ar)
    nb_assets = get_nb_assets(*trial_model.args)
    ar_reshaped = trial_model.args[0]

    t0 = frozen_regression.ppf(0.25).min()
    tf = 10 * frozen_regression.ppf(0.75).max()
    nb_samples = 10

    iterable = NonHomogeneousPoissonProcessIterable(
        nhpp, nb_samples, (t0, tf), ar=ar, seed=21
    )
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # check all times are bounded by the age of replacement
    # add a small constant for numerical approximations
    for i in range(nb_assets):
        ar_asset = np.atleast_1d(ar_reshaped)[i]
        times = select_from_struct(struct_array, asset_id=i)["time"]
        assert (times <= ar_asset + 1e-5).all()
