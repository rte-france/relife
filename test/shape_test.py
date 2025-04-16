import itertools
import random

import numpy as np
import pytest
from numpy.typing import NDArray

from relife.lifetime_model import (
    Weibull,
    Gompertz,
    ProportionalHazard,
    AFT,
    Gamma,
    LogLogistic,
)


"""
distribution : LifetimeDistribution

IN (time)		| OUT
()			 	| ()	# 1 asset, 1 time
(n,)			| (n,)	# 1 asset, n times
(m,1)			| (m,1)	# m assets, 1 time/asset
(m, n)			| (m,n)	# m assets, n times/asset

regression : LifetimeRegression

IN (time, covar)| OUT
(), (k,) 		| ()	# 1 asset, 1 time, broadcasted covar
(n,), (k,) 		| (n,)	# 1 asset, n times, broadcasted covar
(m,1), (k,) 	| (m,1)	# m assets, 1 time/asset, broadcasted covar
(m,n), (k,) 	| (m,n)	# m assets, n times/asset, broadcasted covar
(), (m,k) 		| (m,1)	# m assets, 1 time/asset
(n,), (m,k)	    | (m,n)	# m assets, n times/asset
(m,1), (m,k)	| (m,1)	# m assets, 1 time/asset
(m,n), (m,k) 	| (m,n)	# m assets, n times/asset
(m,n), (z,k) 	| Error # m assets in time, z assets in covar
"""

N = 10
M = 3
K = 5
random.seed(10)
np.random.seed(10)

model_instances = {
    "distribution": [
        Weibull(2, 0.05),
        Gompertz(0.01, 0.1),
        Gamma(2, 0.05),
        LogLogistic(3, 0.05),
    ]
}
model_instances["regression"] = [
    ProportionalHazard(distri, coef=(random.uniform(1.0, 2.0),) * K)
    for distri in model_instances["distribution"]
]
model_instances["regression"] += [
    AFT(distri, coef=(random.uniform(1.0, 2.0),) * K)
    for distri in model_instances["distribution"]
]
model_instances["frozen_distribution"] = [
    model.freeze() for model in model_instances["distribution"]
]
model_instances["frozen_regression"] = [
    model.freeze() for model in model_instances["regression"]
]

io = {
    "distribution": [
        (np.random.uniform(2.5, 10.0, size=()), ()),
        (np.random.uniform(2.5, 10.0, size=(N,)), (N,)),
        (np.random.uniform(2.5, 10.0, size=(M, 1)), (M, 1)),
        (np.random.uniform(2.5, 10.0, size=(M, N)), (M, N)),
    ],
    "regression": [
        (
            np.random.uniform(2.5, 10.0, size=()),
            np.random.uniform(2.5, 10.0, size=(K,)),
            (),
        ),
        (
            np.random.uniform(2.5, 10.0, size=(N,)),
            np.random.uniform(2.5, 10.0, size=(K,)),
            (N,),
        ),
        (
            np.random.uniform(2.5, 10.0, size=(M, 1)),
            np.random.uniform(2.5, 10.0, size=(K,)),
            (M, 1),
        ),
        (
            np.random.uniform(2.5, 10.0, size=(M, N)),
            np.random.uniform(2.5, 10.0, size=(K,)),
            (M, N),
        ),
        (random.uniform(2.5, 10.0), np.random.uniform(2.5, 10.0, size=(M, K)), (M, 1)),
        (
            np.random.uniform(2.5, 10.0, size=(N,)),
            np.random.uniform(2.5, 10.0, size=(M, K)),
            (M, N),
        ),
        (
            np.random.uniform(2.5, 10.0, size=(M, 1)),
            np.random.uniform(2.5, 10.0, size=(M, K)),
            (M, 1),
        ),
        (
            np.random.uniform(2.5, 10.0, size=(M, N)),
            np.random.uniform(2.5, 10.0, size=(M, K)),
            (M, N),
        ),
    ],
}


@pytest.mark.parametrize(
    "model, time, output_shape",
    list(itertools.product(model_instances["distribution"], io["distribution"])),
)
def test_distribution(model, time: NDArray[np.float64], output_shape: tuple[int, ...]):
    assert model.sf(time) == output_shape
    assert model.hf(time) == output_shape
    assert model.chf(time) == output_shape
    assert model.cdf(time) == output_shape
    assert model.pdf(time) == output_shape
    probability = np.random.rand(*time.shape)
    assert model.ppf(probability) == output_shape
    assert model.ichf(probability) == output_shape


@pytest.mark.parametrize(
    "model, time, output_shape",
    list(itertools.product(model_instances["frozen_regression"], io["regression"])),
)
def test_frozen_distribution(
    model, time: NDArray[np.float64], output_shape: tuple[int, ...]
):
    assert model.sf(time) == output_shape
    assert model.hf(time) == output_shape
    assert model.chf(time) == output_shape
    assert model.cdf(time) == output_shape
    assert model.pdf(time) == output_shape
    probability = np.random.rand(*time.shape)
    assert model.ppf(probability) == output_shape
    assert model.ichf(probability) == output_shape


@pytest.mark.parametrize(
    "model, time, covar, output_shape",
    list(itertools.product(model_instances["regression"], io["regression"])),
)
def test_regression(
    model,
    time: NDArray[np.float64],
    covar: NDArray[np.float64],
    output_shape: tuple[int, ...],
):
    assert model.sf(time, covar) == output_shape
    assert model.hf(time, covar) == output_shape
    assert model.chf(time, covar) == output_shape
    assert model.cdf(time, covar) == output_shape
    assert model.pdf(time, covar) == output_shape
    probability = np.random.rand(*time.shape)
    assert model.ppf(probability, covar) == output_shape
    assert model.ichf(probability, covar) == output_shape


@pytest.mark.parametrize(
    "model, time, _, output_shape",
    list(itertools.product(model_instances["frozen_regression"], io["regression"])),
)
def test_frozen_regression(
    model, time: NDArray[np.float64], _, output_shape: tuple[int, ...]
):
    assert model.sf(time) == output_shape
    assert model.hf(time) == output_shape
    assert model.chf(time) == output_shape
    assert model.cdf(time) == output_shape
    assert model.pdf(time) == output_shape
    probability = np.random.rand(*time.shape)
    assert model.ppf(probability) == output_shape
    assert model.ichf(probability) == output_shape
