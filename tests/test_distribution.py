# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import pytest
import numpy as np
from pathlib import Path

from relife.datasets import load_power_transformer
from relife.distribution import (
    Exponential,
    MinimumDistribution,
    Weibull,
    Gompertz,
    Gamma,
    LogLogistic,
)
from relife.data import LifetimeData
# fixtures


@pytest.fixture(scope="module")
def data():
    return load_power_transformer()


@pytest.fixture(
    scope="module",
    params=[
        Exponential(0.00795203),
        Weibull(3.46597395, 0.01227849),
        Gompertz(0.00865741, 0.06062632),
        Gamma(5.3571091, 0.06622822),
        LogLogistic(3.92614064, 0.0133325),
    ],
)
def model(request):
    return request.param


# test functions


def test_sf(model):
    assert model.sf(model.median()) == pytest.approx(0.5, rel=1e-3)


def test_rvs(model):
    size = 10
    assert model.rvs(size=size).shape == (size,)


def test_mean(model):
    assert super(type(model), model).mean() == pytest.approx(model.mean(), rel=1e-3)


def test_mrl(model):
    t = np.arange(10)
    assert model.mrl(t).shape == (t.size,)

def test_fit(model, data):
    params = model.params.copy()
    model.fit(*data.astuple())
    assert model.params == pytest.approx(params, rel=1e-3)

def test_minimum_distribution(model, data):
    params = model.params.copy()
    n = np.ones((data.size, 1))
    model = MinimumDistribution(model).fit(*data.astuple(), args=(n,))
    assert model.params == pytest.approx(params, rel=1e-3)

def test_2D_data(model):
    DATA_PATH = Path(__file__).parent.parent / "relife" / "datasets"
    data = np.loadtxt(
        DATA_PATH /"power_transformer.csv", delimiter=",", skiprows=1, unpack=True
    )
    # return LifetimeData(*data)
    time, event, entry = data
    ind = int(sum(event))
    exact_obs = np.tile(time[:ind], (2, 1)).T
    lc_obs = np.column_stack((time[ind:], np.inf * np.ones(len(time[ind:]))))
    time_2D = np.vstack((exact_obs, lc_obs))
    assert (time_2D, event, entry) == LifetimeData(time_2D, event, entry).astuple()
    data = LifetimeData(time_2D, event, entry)
    params = model.params.copy()

    model.fit(*data.astuple())
    
    assert model.params == pytest.approx(params, rel=1e-3)

