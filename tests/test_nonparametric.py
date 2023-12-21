# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import pytest
import numpy as np

from relife.datasets import load_power_transformer
from relife.nonparametric import ECDF, KaplanMeier, Turnbull
from relife.data import LifetimeData


# fixtures


@pytest.fixture
def data():
    return load_power_transformer()


# test functions


def test_fit_ecdf_kaplan_meier(data):
    print(data) # TODO : check what's the return of this
    time = data.time[data.event == 1]
    ecdf = ECDF().fit(time)
    km = KaplanMeier().fit(time)
    assert ecdf.sf == pytest.approx(km.sf, rel=1e-4)
    assert False # TODO : remove when done

def test_turnbull():
    # tb = Turnbull().fit
    # TODO : after implementing load_input_turnbull(), checking what data of test_fit_ecdf_kaplan_meier() is, implement same for test_turnbull()
    assert False # TODO : remove when done