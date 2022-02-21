# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import pytest
import numpy as np
from scipy.stats import boxcox, zscore

from relife.datasets import load_insulator_string
from relife.distribution import Exponential, Weibull, Gompertz, Gamma, LogLogistic
from relife.regression import AFT, ProportionalHazards


# fixtures


@pytest.fixture(scope="module")
def data():
    time, event, entry, *args = load_insulator_string().astuple()
    covar = zscore(np.column_stack([boxcox(col)[0] for col in args[0].T]))
    return time, event, entry, covar


@pytest.fixture(
    scope="module",
    params=[
        Exponential(0.05),
        Weibull(2, 0.05),
        Gompertz(0.01, 0.1),
        Gamma(2, 0.05),
        LogLogistic(3, 0.05),
    ],
)
def baseline(request):
    return request.param


@pytest.fixture(scope="module", params=[AFT, ProportionalHazards])
def model(request, baseline):
    beta = [1.0, 1.0]
    return request.param(baseline, beta)


@pytest.fixture(scope="module")
def covar():
    return np.random.uniform(size=(5, 2))


# test functions


def test_sf(model, covar):
    assert model.sf(model.median(covar), covar) == pytest.approx(0.5, rel=1e-3)


def test_rvs(model, covar):
    size = 10
    assert model.rvs(covar, size=size).shape == (covar.shape[0], size)


def test_mean(model, covar):
    assert model.mean(covar).shape[0] == covar.shape[0]


def test_fit(model, data):
    model.fit(*data)


def test_aft_pph_weibull_eq(data):
    model_aft = AFT(Weibull()).fit(*data)
    model_pph = ProportionalHazards(Weibull()).fit(*data)
    assert model_pph.baseline.params == pytest.approx(
        model_aft.baseline.params, rel=1e-3
    )
    assert model_pph.beta == pytest.approx(
        -model_aft.baseline.c * model_aft.beta, rel=1e-3
    )
