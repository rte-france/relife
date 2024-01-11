# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import pytest
import numpy as np

from relife.datasets import load_power_transformer, load_input_turnbull
from relife.nonparametric import ECDF, KaplanMeier, Turnbull
from relife.data import LifetimeData
from relife.distribution import Weibull


@pytest.fixture
def data():
    return load_power_transformer()

@pytest.fixture
def data_turnbull():
    return load_input_turnbull()

@pytest.fixture(
    scope="module",
    params= [ Weibull(c=7.531, rate=0.037)],
)

def weibull_model(request):
    return request.param

# test functions


def test_fit_ecdf_kaplan_meier(data):
    # print(data) # TODO : check what's the return of this
    time = data.time[data.event == 1]
    ecdf = ECDF().fit(time)
    km = KaplanMeier().fit(time)
    assert ecdf.sf == pytest.approx(km.sf, rel=1e-4)
    # assert False # TODO : remove when done

def test_turnbull(data_turnbull, weibull_model):
    import pandas as pd 
    # tb = Turnbull().fit
    # TODO : after implementing load_input_turnbull(), checking what data of test_fit_ecdf_kaplan_meier() is, implement same for test_turnbull()
     
    tb = Turnbull().fit(data_turnbull.time, entry = data_turnbull.entry)
    # print(tb.params)
    print(weibull_model.params)
    # TODO : assert tb estimation equals weibull_model estimation
    assert False 


