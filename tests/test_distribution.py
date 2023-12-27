# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import pytest
import numpy as np

from relife.datasets import load_power_transformer
from relife.distribution import (
    Exponential,
    MinimumDistribution,
    Weibull,
    Gompertz,
    Gamma,
    LogLogistic,
)

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
    import os 
    filename = "output_debugDF.txt"
    name = os.path.join('..', filename)
    
    f = open(name, "w")
    for i in range(len(data._time.D)) : 
        f.write(str(data._time.D[i]))
        f.write("\n")
    
    for i in range(len(data._time.D_RC)) :
        f.write(str(data._time.D_RC[i]))
        f.write("\n")
    
    for i in range(len(data._time.LT)) :
        f.write(str(data._time.LT[i]))
        f.write("\n")

    for i in range(len(data._time.LC)) :
        f.write(str(data._time.LC[i]))
        f.write("\n")
    f.close()

    # print(model)
    # print(data.astuple())
    model.fit(*data.astuple())
    # print(model)
    fDF = open(name, "r")
    filenameTB = "output_debugTB.txt"
    nameTB = os.path.join('..', filenameTB)
    fTB = open(nameTB, "r")
    for i in range(len(data.time)):
        line = fDF.readline()
        lineTB = fTB.readline()
        # print(line)
        # print(lineTB)
        if line != lineTB :
            print("Error in line ", i)
        assert line == lineTB
    i = 1 / np.median(data.time)
    # print(i)
    assert model.params == pytest.approx(params, rel=1e-3)


def test_minimum_distribution(model, data):
    params = model.params.copy()
    n = np.ones((data.size, 1))
    model = MinimumDistribution(model).fit(*data.astuple(), args=(n,))
    assert model.params == pytest.approx(params, rel=1e-3)
