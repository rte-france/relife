"""
distribution : LifetimeDistribution

IN (time)		| OUT
()			 	| ()	# 1 asset, 1 time
(n,)			| (n,)	# 1 asset, n times
(m,1)			| (m,1)	# m assets, 1 time/asset
(m, n)			| (m,n)	# m assets, n times/asset
"""

import pytest
from pytest import approx
import numpy as np


@pytest.mark.parametrize("fixture_name", ["weibull", "gompertz", "gamma", "loglogistic"])
def test_distribution(distribution_map, fixture_name, time, probability):
    distribution = distribution_map[fixture_name]

    assert distribution.moment(1).shape == ()
    assert distribution.moment(2).shape == ()
    assert distribution.mean().shape == ()
    assert distribution.var().shape == ()
    assert distribution.median().shape == ()

    assert distribution.sf(time()).shape == ()
    assert distribution.hf(time()).shape == ()
    assert distribution.chf(time()).shape == ()
    assert distribution.cdf(time()).shape == ()
    assert distribution.pdf(time()).shape == ()
    assert distribution.ppf(probability()).shape == ()
    assert distribution.rvs(1, seed=21).shape == ()
    assert distribution.ichf(probability()).shape == ()
    assert distribution.isf(probability()).shape == ()
    assert distribution.isf(0.5) == approx(distribution.median())
    assert distribution.dhf(time()).shape == ()
    assert distribution.jac_sf(time()).shape == (1, 2)
    assert distribution.jac_hf(time()).shape == (1, 2)
    assert distribution.jac_chf(time()).shape == (1, 2)
    assert distribution.jac_cdf(time()).shape == (1, 2)
    assert distribution.jac_pdf(time()).shape == (1, 2)

    n = 10

    assert distribution.sf(time(n)).shape == (n,)
    assert distribution.hf(time(n)).shape == (n,)
    assert distribution.chf(time(n)).shape == (n,)
    assert distribution.cdf(time(n)).shape == (n,)
    assert distribution.pdf(time(n)).shape == (n,)
    assert distribution.ppf(
        probability(
            n,
        )
    ).shape == (n,)
    assert distribution.rvs((n,), seed=21).shape == (n,)
    assert distribution.ichf(
        probability(
            n,
        )
    ).shape == (n,)
    assert distribution.isf(
        probability(
            n,
        )
    ).shape == (n,)
    assert distribution.isf(np.full((n,), 0.5)) == approx(np.full((n,), distribution.median()))
    assert distribution.dhf(
        time(
            n,
        )
    ).shape == (n,)
    assert distribution.jac_sf(
        time(
            n,
        )
    ).shape == (n, 2)
    assert distribution.jac_hf(
        time(
            n,
        )
    ).shape == (n, 2)
    assert distribution.jac_chf(
        time(
            n,
        )
    ).shape == (n, 2)
    assert distribution.jac_cdf(
        time(
            n,
        )
    ).shape == (n, 2)
    assert distribution.jac_pdf(
        time(
            n,
        )
    ).shape == (n, 2)

    m = 3

    assert distribution.sf(time(m, 1)).shape == (m, 1)
    assert distribution.hf(time(m, 1)).shape == (m, 1)
    assert distribution.chf(time(m, 1)).shape == (m, 1)
    assert distribution.cdf(time(m, 1)).shape == (m, 1)
    assert distribution.pdf(time(m, 1)).shape == (m, 1)
    assert distribution.ppf(probability(m, 1)).shape == (m, 1)
    assert distribution.rvs((m, 1), seed=21).shape == (m, 1)
    assert distribution.ichf(probability(m, 1)).shape == (m, 1)
    assert distribution.isf(probability(m, 1)).shape == (m, 1)
    assert distribution.isf(np.full((m, 1), 0.5)) == approx(np.full((m, 1), distribution.median()))
    assert distribution.dhf(time(m, 1)).shape == (m, 1)
    assert distribution.jac_sf(time(m, 1)).shape == (m, 2)
    assert distribution.jac_hf(time(m, 1)).shape == (m, 2)
    assert distribution.jac_chf(time(m, 1)).shape == (m, 2)
    assert distribution.jac_cdf(time(m, 1)).shape == (m, 2)
    assert distribution.jac_pdf(time(m, 1)).shape == (m, 2)

    assert distribution.sf(time(m, n)).shape == (m, n)
    assert distribution.hf(time(m, n)).shape == (m, n)
    assert distribution.chf(time(m, n)).shape == (m, n)
    assert distribution.cdf(time(m, n)).shape == (m, n)
    assert distribution.pdf(time(m, n)).shape == (m, n)
    assert distribution.ppf(probability(m, n)).shape == (m, n)
    assert distribution.rvs((m, n), seed=21).shape == (m, n)
    assert distribution.ichf(probability(m, n)).shape == (m, n)
    assert distribution.isf(probability(m, n)).shape == (m, n)
    assert distribution.isf(np.full((m, n), 0.5)) == approx(np.full((m, n), distribution.median()))
    assert distribution.dhf(time(m, n)).shape == (m, n)
    with pytest.raises(ValueError) as err:
        distribution.jac_sf(time(m, n))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
    with pytest.raises(ValueError) as err:
        distribution.jac_hf(time(m, n))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
    with pytest.raises(ValueError) as err:
        distribution.jac_chf(time(m, n))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
    with pytest.raises(ValueError) as err:
        distribution.jac_pdf(time(m, n))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
    with pytest.raises(ValueError) as err:
        distribution.jac_cdf(time(m, n))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
