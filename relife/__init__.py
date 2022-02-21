"""The relife package."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

from .model import AgeReplacementModel, LeftTruncated, EquilibriumDistribution
from .distribution import (
    Exponential,
    Weibull,
    Gompertz,
    Gamma,
    LogLogistic,
    MinimumDistribution,
)
from .regression import AFT, ProportionalHazards
from .nonparametric import ECDF, KaplanMeier, NelsonAalen
from .renewal_process import RenewalProcess, RenewalRewardProcess
from .replacement_policy import (
    OneCycleRunToFailure,
    RunToFailure,
    OneCycleAgeReplacementPolicy,
    AgeReplacementPolicy,
)

__version__ = "1.0.0"

__all__ = [
    "AgeReplacementModel",
    "LeftTruncated",
    "EquilibriumDistribution",
    "Exponential",
    "Weibull",
    "Gompertz",
    "Gamma",
    "LogLogistic",
    "MinimumDistribution",
    "AFT",
    "ProportionalHazards",
    "ECDF",
    "KaplanMeier",
    "NelsonAalen",
    "RenewalProcess",
    "RenewalRewardProcess",
    "OneCycleRunToFailure",
    "RunToFailure",
    "OneCycleAgeReplacementPolicy",
    "AgeReplacementPolicy",
]
