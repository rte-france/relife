# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import pytest
import numpy as np
from itertools import chain

from relife.replacement_policy import (
    OneCycleRunToFailure,
    RunToFailure,
    OneCycleAgeReplacementPolicy,
    AgeReplacementPolicy,
)
from relife.utils import args_size
from .test_renewal_process import baseline


# fixtures


@pytest.fixture(scope="module", params=[0, 0.04])
def fit_args(request):
    rate = request.param
    cp = 1
    cf = cp + np.array([5, 10, 20, 100, 1000]).reshape(-1, 1)
    return cf, cp, rate


@pytest.fixture(
    scope="module",
    params=[
        OneCycleRunToFailure,
        OneCycleAgeReplacementPolicy,
        RunToFailure,
        AgeReplacementPolicy,
    ],
)
def policy(request, baseline, fit_args):
    policy = request.param(baseline)
    cf, cp, rate = fit_args
    policy.cf = cf
    policy.rate = rate
    if hasattr(policy, "cp"):
        policy.cp = cp
        policy.fit()
    return policy


# test functions


def test_one_cycle_optimal_replacement_age(baseline, fit_args):
    cf, cp, rate = fit_args
    eps = 1e-2
    policy = OneCycleAgeReplacementPolicy(baseline, cf=cf, cp=cp, rate=rate).fit()
    ar = policy.ar
    assert np.all(
        policy.asymptotic_expected_equivalent_annual_cost(ar + eps, dt=0.1)
        > policy.asymptotic_expected_equivalent_annual_cost()
    ) and np.all(
        policy.asymptotic_expected_equivalent_annual_cost(ar - eps, dt=0.1)
        > policy.asymptotic_expected_equivalent_annual_cost()
    )


def test_optimal_replacement_age(baseline, fit_args):
    cf, cp, rate = fit_args
    eps = 1e-2
    policy = AgeReplacementPolicy(baseline, cf=cf, cp=cp, rate=rate).fit(fit_ar1=False)
    ar = policy.ar
    assert np.all(
        policy.asymptotic_expected_equivalent_annual_cost(ar + eps)
        > policy.asymptotic_expected_equivalent_annual_cost()
    ) and np.all(
        policy.asymptotic_expected_equivalent_annual_cost(ar - eps)
        > policy.asymptotic_expected_equivalent_annual_cost()
    )


def test_asymptotic_expected_equivalent_annual_cost(policy):
    t = np.arange(0, 400, 0.2)
    qa = policy.asymptotic_expected_equivalent_annual_cost()
    q = policy.expected_equivalent_annual_cost(t)
    assert q[..., -1:] == pytest.approx(qa, rel=1e-1)


def test_expected_total_cost_vec(policy):
    n = 3
    policy.cf = np.tile(policy.cf, (n, 1, 1))
    policy.rate = np.tile(policy.rate, (n, 1, 1))
    t = np.arange(0, 100, 0.5)
    z = policy.expected_total_cost(t)
    assert z.sum(axis=0) == pytest.approx(n * z[0, ...], rel=1e-4)


def test_sample(policy):
    n_indices = max(1, args_size(*chain(*policy.rrp_args())))
    n_samples = 10
    if hasattr(policy, "rrp"):
        data = policy.sample(0, n_samples=n_samples)
    else:
        data = policy.sample(n_samples=n_samples)
    assert data.size == n_indices * n_samples
