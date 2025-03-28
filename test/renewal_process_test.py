# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import numpy as np
import pytest

from relife.distributions import (
    AFT,
    Gamma,
    Gompertz,
    LogLogistic,
    ProportionalHazard,
    Weibull,
)
from relife.economics.rewards import run_to_failure_rewards
from relife.parametric.composition import (
    AgeReplacementDistribution,
    EquilibriumDistribution,
)
from relife.processes import RenewalProcess, RenewalRewardProcess


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


@pytest.fixture(scope="module", params=[None, AFT, ProportionalHazard])
def regression(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        None,
        AgeReplacementDistribution,
    ],
)
def age_replacement_model(request):
    return request.param


@pytest.fixture(scope="module")
def model_args_nb_assets(baseline, regression, age_replacement_model):
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
        if not isinstance(tmax, np.ndarray):
            tmax = np.array(tmax)
        args = (tmax,) + args
    nb_assets = max(
        tuple(map(lambda x: x.shape[0] if x.ndim > 1 else 1, args)), default=1
    )
    return model, args, nb_assets


# test functions


def test_renewal_process(model_args_nb_assets):
    t = np.arange(0, 100, 0.5)
    model, model_args, nb_assets = model_args_nb_assets
    model1 = EquilibriumDistribution(model)
    rp = RenewalProcess(
        model,
        model1=model1,
        model_args=model_args,
        model1_args=model_args,
        nb_assets=nb_assets,
    )
    y0 = 1 / model.mean(*rp.model_args)
    y = rp.renewal_density(t)
    assert y[..., -1:] == pytest.approx(y0, rel=1e-4)


def test_renewal_reward_process(model_args_nb_assets):
    t = np.arange(0, 100, 0.5)
    model, model_args, nb_assets = model_args_nb_assets
    reward = run_to_failure_rewards(cf=1)
    rrp = RenewalRewardProcess(
        model,
        reward,
        model_args=model_args,
        nb_assets=nb_assets,
    )
    m = rrp.renewal_function(t)
    z = rrp.expected_total_reward(t)
    assert m == pytest.approx(z, rel=1e-4)


def test_renewal_reward_process_vec(model_args_nb_assets):
    t = np.arange(0, 100, 0.5)
    cf0 = 1
    discounting_rate = 0.04
    model, model_args, nb_assets = model_args_nb_assets

    nb_assets = max(
        tuple(map(lambda x: x.shape[0] if x.ndim >= 1 else 1, model_args)), default=1
    )
    n = (
        5 if nb_assets == 1 else nb_assets
    )  # vectorizes in 5 assets else equals the number of assets (could be a float too)
    cf = cf0 / n

    rrp0 = RenewalRewardProcess(
        model,
        run_to_failure_rewards(cf=cf0),
        model_args=model_args,
        discounting_rate=discounting_rate,
        nb_assets=nb_assets,
    )
    rrp = RenewalRewardProcess(
        model,
        run_to_failure_rewards(cf=np.full((n, 1), cf)),
        model_args=model_args,
        discounting_rate=discounting_rate,
        nb_assets=n,
    )
    z0 = rrp0.expected_total_reward(t)
    z = rrp.expected_total_reward(t)
    # if one asset, then z has 2 dim with n lines of expected_total_reward
    if nb_assets == 1:
        assert z0 == pytest.approx(z.sum(axis=0), rel=1e-4)
    # if assets, then z0 has already nb_assets lines of expected_total_reward on first dim
    else:
        assert z0 == pytest.approx(n * z, rel=1e-4)
