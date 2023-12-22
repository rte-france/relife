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

# fixtures [aya : are these needed? using only once in test_fit_ecdf_kaplan_meier() and test_turnbull() ]

@pytest.fixture
def data():
    return load_power_transformer()

@pytest.fixture
def data_turnbull():
    return load_input_turnbull()

# test functions


def test_fit_ecdf_kaplan_meier(data):
    print(data) # TODO : check what's the return of this
    time = data.time[data.event == 1]
    ecdf = ECDF().fit(time)
    km = KaplanMeier().fit(time)
    assert ecdf.sf == pytest.approx(km.sf, rel=1e-4)
    # assert False # TODO : remove when done

def test_turnbull(data_turnbull):
    # tb = Turnbull().fit
    # TODO : after implementing load_input_turnbull(), checking what data of test_fit_ecdf_kaplan_meier() is, implement same for test_turnbull()
    
    print(data_turnbull.entry)
    print(data_turnbull.time)
    data = np.column_stack((data_turnbull.time, data_turnbull.entry))
    print(data)
    time2d = np.array([[1,2],[0,4],[5,5],[7,np.inf],[10,10], [2,10], [10,11]])
    l = LifetimeData(time2d)
    print(l.xl)
    timeline, inv, counts = np.unique(
        l.xl, return_inverse=True, return_counts=True
    )
    print(timeline, inv, counts)
    print(l.event)
    d = np.zeros_like(timeline, int)
    print(d)
    np.add.at(d, inv, l.event)
    print(d)
    # d = np.zeros_like(timeline, int)
    # np.add.at(d, inv, data.event)
    assert False # TODO : remove when done

