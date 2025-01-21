# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import numpy as np
import pytest

from relife2 import (
    OneCycleRunToFailure,
    RunToFailure,
    OneCycleAgeReplacementPolicy,
    AgeReplacementPolicy,
)
from relife2 import Weibull, Gompertz, Gamma, LogLogistic


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


@pytest.fixture(scope="module", params=[0, 0.04])
def fit_args(request):
    discount_rate = request.param
    cp = 1
    cf = cp + np.array([5, 10, 20, 100, 1000]).reshape(-1, 1)
    return cf, cp, discount_rate


@pytest.fixture(
    scope="module",
    params=[
        OneCycleRunToFailure,
        OneCycleAgeReplacementPolicy,
        RunToFailure,
    ],
)
def policy(request, baseline, fit_args):
    cf, cp, discount_rate = fit_args

    if request.param == OneCycleRunToFailure or request.param == RunToFailure:
        policy_obj = request.param(
            baseline, cf, discount_rate=discount_rate, nb_assets=5
        )
    else:
        policy_obj = request.param(baseline, cf, cp, discount_rate, nb_assets=5)
        policy_obj.fit(inplace=True)

    return policy_obj


@pytest.fixture(
    scope="module",
    params=[
        OneCycleRunToFailure,
        OneCycleAgeReplacementPolicy,
        RunToFailure,
    ],
)
def policy_vec(request, baseline, fit_args):
    cf, cp, discount_rate = fit_args
    batch_size = 3
    cf = np.tile(cf, (batch_size, 1, 1))
    discount_rate = np.tile(discount_rate, (batch_size, 1, 1))

    if request.param == OneCycleRunToFailure or request.param == RunToFailure:
        policy_obj = request.param(baseline, cf, discount_rate, nb_assets=5)
    else:
        policy_obj = request.param(baseline, cf, cp, discount_rate, nb_assets=5)
        policy_obj.fit(inplace=True)
    return policy_obj


# test functions


def test_one_cycle_optimal_replacement_age(baseline, fit_args):
    cf, cp, discount_rate = fit_args
    eps = 1e-2
    policy = OneCycleAgeReplacementPolicy(
        baseline, cf, cp, discount_rate=discount_rate, nb_assets=5
    )
    policy.fit(inplace=True)
    ar = policy.ar

    policy1 = OneCycleAgeReplacementPolicy(
        baseline, cf, cp, discount_rate=discount_rate, ar=ar + eps, nb_assets=5
    )
    policy0 = OneCycleAgeReplacementPolicy(
        baseline, cf, cp, discount_rate=discount_rate, ar=ar - eps, nb_assets=5
    )

    policy.asymptotic_expected_equivalent_annual_cost()

    assert np.all(
        policy1.asymptotic_expected_equivalent_annual_cost(dt=0.1)
        > policy.asymptotic_expected_equivalent_annual_cost()
    ) and np.all(
        policy0.asymptotic_expected_equivalent_annual_cost(dt=0.1)
        > policy.asymptotic_expected_equivalent_annual_cost()
    )


def test_optimal_replacement_age(baseline, fit_args):
    cf, cp, discount_rate = fit_args
    eps = 1e-2
    policy = AgeReplacementPolicy(baseline, cf, cp, discount_rate=discount_rate).fit()
    ar = policy.ar

    policy1 = AgeReplacementPolicy(
        baseline, cf, cp, discount_rate=discount_rate, ar=ar + eps
    )
    policy0 = AgeReplacementPolicy(
        baseline, cf, cp, discount_rate=discount_rate, ar=ar - eps
    )

    assert np.all(
        policy1.asymptotic_expected_equivalent_annual_cost()
        > policy.asymptotic_expected_equivalent_annual_cost()
    ) and np.all(
        policy0.asymptotic_expected_equivalent_annual_cost()
        > policy.asymptotic_expected_equivalent_annual_cost()
    )


def test_asymptotic_expected_equivalent_annual_cost(policy):
    timeline = np.arange(0, 400, 0.2)
    qa = policy.asymptotic_expected_equivalent_annual_cost()
    q = policy.expected_equivalent_annual_cost(timeline)
    assert q[..., -1:] == pytest.approx(qa, rel=1e-1)


# FIXME : does not work because now max ndim in ls_integrate is 2d, here it 3d -> broadcasting error
# possible solutions :
# 1. set ndim (ReLife 1)
# 2. skip this test (obsolete)
# 3. create a more complex mecanism to infer ndim inside ls_integrate
# def test_expected_total_cost_vec(policy_vec):
#     batch_size = 3
#     timeline = np.arange(0, 100, 0.5)
#     z = policy_vec.expected_total_cost(timeline)
#     assert z.sum(axis=0) == pytest.approx(batch_size * z[0, ...], rel=1e-4)


# FIXME : if end_time == 0., always atleast one sample was returned in old ReLife.
# just need to change still_valid update order in sample_routine ?
def test_sample(policy):
    nb_assets = 5  # supposed to be set at initialization
    nb_samples = 10
    if isinstance(policy, RunToFailure):
        data = policy.sample(nb_samples, 0.0)
    else:
        data = policy.sample(nb_samples)
    assert len(data) == nb_samples * nb_assets
