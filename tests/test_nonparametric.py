# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import pytest
import numpy as np

from relife.datasets import load_power_transformer
from relife.nonparametric import ECDF, KaplanMeier
from relife.data import LifetimeData

# fixtures


@pytest.fixture
def data():
    return load_power_transformer()


# test functions


def test_fit_ecdf_kaplan_meier(data):
    time = data.time[data.event == 1]
    ecdf = ECDF().fit(time)
    km = KaplanMeier().fit(time)
    assert ecdf.sf == pytest.approx(km.sf, rel=1e-4)

def test_data_format_2D():
    """
    Fonction pour visualiser en faisant un print des resultats en attendant de fr des test unitaires
    """
    print("testing...")
    time2d = np.array([[1,2],[0,4],[5,5],[7,np.inf],[10,10], [2,10], [10,11]])
    
    l = LifetimeData(time2d)
    assert l.time.shape == (7,2)
    # assert False

