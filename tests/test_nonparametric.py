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
    time = data.time[data.event == 1]
    ecdf = ECDF().fit(time)
    km = KaplanMeier().fit(time)
    assert ecdf.sf == pytest.approx(km.sf, rel=1e-4)

def test_turnbull(data_turnbull, weibull_model):
    """
    Test the Turnbull estimator.

    Explications : 
        - Lors de la création de data_turnbull, on s'est basé sur Weibull(c=7.531, rate=0.037) 
    pour déterminer si un interval contenait une defaillance ou non. 

        - Division de tb.timeline par 6 : On avait multiplié les temps de Weibull(c=7.531, rate=0.037) 
    par 6 pour avoir des valeurs cohérente avec les valeurs des intervales de visite de poste.

    """
    tb = Turnbull().fit(data_turnbull.time, entry = data_turnbull.entry)
    t = tb.timeline / 6
    assert np.isclose(tb.sf, weibull_model.sf(t), atol=0.02).all()
    # S = np.linspace(0, 1, 163)
    # print(np.linspace(0, 1, 163))
    # print(np.diff(S))
    # assert False 
