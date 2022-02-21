# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import pytest
import numpy as np

from relife.model import AgeReplacementModel, EquilibriumDistribution
from relife.distribution import Weibull, Gompertz, Gamma, LogLogistic
from relife.regression import AFT, ProportionalHazards
from relife.renewal_process import RenewalProcess, RenewalRewardProcess
from relife.reward import FailureCost

# fixtures


@pytest.fixture(
    scope="module",
    params=[
        Weibull(2, 0.05),
        Gompertz(0.01, 0.1),
        Gamma(2, 0.05),
        LogLogistic(3, 0.05),
    ],
)
def baseline(request):
    return request.param


@pytest.fixture(scope="module", params=[None, AFT, ProportionalHazards])
def regression(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        None,
        AgeReplacementModel,
    ],
)
def age_replacement_model(request):
    return request.param


@pytest.fixture(scope="module")
def model_and_args(baseline, regression, age_replacement_model):
    model = baseline
    args = ()
    if regression is not None:
        beta = [np.log(2), np.log(2)]
        covar = np.arange(0.0, 0.6, 0.1).reshape(-1, 2)
        model = regression(model, beta)
        args = (covar,) + args
    if age_replacement_model is not None:
        tmax = model.isf(0.75, *args)
        model = age_replacement_model(model)
        args = (tmax,) + args
    return model, args


# test functions


def test_renewal_process(model_and_args):
    t = np.arange(0, 100, 0.5)
    model, model_args = model_and_args
    model1 = EquilibriumDistribution(model)
    rp = RenewalProcess(model, model1)
    y0 = 1 / model.mean(*model_args)
    y = rp.renewal_density(t, model_args, model_args)
    assert y[..., -1:] == pytest.approx(y0, rel=1e-4)


def test_renewal_reward_process(model_and_args):
    t = np.arange(0, 100, 0.5)
    model, model_args = model_and_args
    reward = FailureCost()
    reward_args = (1,)
    rrp = RenewalRewardProcess(model, reward)
    m = rrp.renewal_function(t, model_args)
    z = rrp.expected_total_reward(t, model_args, reward_args)
    assert m == pytest.approx(z, rel=1e-4)


def test_renewal_reward_process_vec(model_and_args):
    t = np.arange(0, 100, 0.5)
    n = 3
    cf = 1
    rate = 0.04
    model, model_args = model_and_args
    reward = FailureCost()
    reward_args0 = (n * cf,)
    reward_args = (cf,)
    discount_args0 = (rate,)
    discount_args = (np.repeat(rate, n).reshape(-1, 1, 1),)
    rrp = RenewalRewardProcess(model, reward)
    z0 = np.atleast_2d(
        rrp.expected_total_reward(
            t, model_args, reward_args0, discount_args=discount_args0
        )
    )
    z = rrp.expected_total_reward(
        t, model_args, reward_args, discount_args=discount_args
    )
    assert z0 == pytest.approx(z.sum(axis=0), rel=1e-4)
