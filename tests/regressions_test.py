"""
Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
This file is part of ReLife, an open source Python library for asset
management based on reliability theory and lifetime data analysis.
"""

import numpy as np
import pytest
from scipy.stats import boxcox, zscore

from datasets import load_insulator_string
from relife2 import (
    AFT,
    ProportionalHazard,
    LogLogistic,
    Exponential,
    Weibull,
    Gompertz,
    Gamma,
)


# fixtures


@pytest.fixture(scope="module")
def data():
    return load_insulator_string()


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


@pytest.fixture(scope="module", params=[AFT, ProportionalHazard])
def model(request, baseline):
    return request.param(baseline, (1.0, 1.0, 1.0))


@pytest.fixture(scope="module")
def covar():
    return np.random.uniform(size=(5, 3))


@pytest.fixture(scope="module")
def weibull_aft():
    return AFT(Weibull())


@pytest.fixture(scope="module")
def weibull_pph():
    return ProportionalHazard(Weibull())


# test functions


def test_sf(model, covar):
    assert model.sf(model.median(covar), covar) == pytest.approx(0.5, rel=1e-3)


def test_rvs(model, covar):
    size = 10
    assert model.rvs(covar, size=size).shape == (covar.shape[0], size)


def test_mean(model, covar):
    assert model.mean(covar).shape[0] == covar.shape[0]


def fit_model(model, data):
    model.fit(
        data[0, :],
        event=data[1, :] == 1,
        entry=data[2, :],
        args=(
            zscore(
                np.column_stack(
                    [boxcox(covar_values)[0] for covar_values in data[3:, :]]
                )
            ),
        ),
    )
    return model


def test_fit_model(model, data):
    fit_model(model, data)


def test_aft_pph_weibull_eq(data, weibull_aft, weibull_pph):

    weibull_aft = fit_model(weibull_aft, data)
    weibull_pph = fit_model(weibull_pph, data)

    assert weibull_pph.baseline.params == pytest.approx(
        weibull_aft.baseline.params, rel=1e-3
    )
    assert weibull_pph.covar_effect.params == pytest.approx(
        -weibull_aft.baseline.shape * weibull_aft.covar_effect.params,
        rel=1e-3,
    )
