import pytest
from pytest import approx
import numpy as np

from relife.lifetime_model import EquilibriumDistribution

def test_args_names(distribution):
    assert distribution.args_names == ()
    assert EquilibriumDistribution(distribution).args_names == ()

def test_rvs(distribution):
    m, n = 3, 10
    assert distribution.rvs(1, seed=21).shape == ()
    assert distribution.rvs((n,), seed=21).shape == (n,)
    assert distribution.rvs((m, 1), seed=21).shape == (m, 1)
    assert distribution.rvs((m, n), seed=21).shape == (m, n)

def test_probility_functions(distribution, time, probability):
    m, n = 3, 10

    assert distribution.sf(time()).shape == ()
    assert distribution.sf(distribution.median()) == approx(0.5, rel=1e-3)
    assert distribution.hf(time()).shape == ()
    assert distribution.chf(time()).shape == ()
    assert distribution.cdf(time()).shape == ()
    assert distribution.pdf(time()).shape == ()
    assert distribution.ppf(probability()).shape == ()
    assert distribution.ichf(probability()).shape == ()
    assert distribution.isf(probability()).shape == ()
    assert distribution.isf(0.5) == approx(distribution.median())

    assert distribution.sf(time(n)).shape == (n,)
    assert distribution.sf(np.full((n,), distribution.median())) == approx(np.full((n,), 0.5), rel=1e-3)
    assert distribution.hf(time(n)).shape == (n,)
    assert distribution.chf(time(n)).shape == (n,)
    assert distribution.cdf(time(n)).shape == (n,)
    assert distribution.pdf(time(n)).shape == (n,)
    assert distribution.ppf(probability(n,)).shape == (n,)
    assert distribution.ichf(probability(n,)).shape == (n,)
    assert distribution.isf(probability(n,)).shape == (n,)
    assert distribution.isf(np.full((n,), 0.5)) == approx(np.full((n,), distribution.median()))

    assert distribution.sf(time(m, 1)).shape == (m, 1)
    assert distribution.sf(np.full((m, 1), distribution.median())) == approx(np.full((m, 1), 0.5), rel=1e-3)
    assert distribution.hf(time(m, 1)).shape == (m, 1)
    assert distribution.chf(time(m, 1)).shape == (m, 1)
    assert distribution.cdf(time(m, 1)).shape == (m, 1)
    assert distribution.pdf(time(m, 1)).shape == (m, 1)
    assert distribution.ppf(probability(m, 1)).shape == (m, 1)
    assert distribution.ichf(probability(m, 1)).shape == (m, 1)
    assert distribution.isf(probability(m, 1)).shape == (m, 1)
    assert distribution.isf(np.full((m, 1), 0.5)) == approx(np.full((m, 1), distribution.median()))

    assert distribution.sf(time(m, n)).shape == (m, n)
    assert distribution.sf(np.full((m, n), distribution.median())) == approx(np.full((m, n), 0.5), rel=1e-3)
    assert distribution.hf(time(m, n)).shape == (m, n)
    assert distribution.chf(time(m, n)).shape == (m, n)
    assert distribution.cdf(time(m, n)).shape == (m, n)
    assert distribution.pdf(time(m, n)).shape == (m, n)
    assert distribution.ppf(probability(m, n)).shape == (m, n)
    assert distribution.ichf(probability(m, n)).shape == (m, n)
    assert distribution.isf(probability(m, n)).shape == (m, n)
    assert distribution.isf(np.full((m, n), 0.5)) == approx(np.full((m, n), distribution.median()))


def test_moment(distribution, time):
    assert distribution.moment(1).shape == ()
    assert distribution.moment(2).shape == ()
    assert distribution.mean().shape == ()
    assert distribution.var().shape == ()
    assert distribution.median().shape == ()


def test_derivative(distribution, time):
    m, n = 3, 10

    assert distribution.dhf(time()).shape == ()
    assert distribution.jac_sf(time()).shape == (distribution.nb_params,)
    assert distribution.jac_hf(time()).shape == (distribution.nb_params,)
    assert distribution.jac_chf(time()).shape == (distribution.nb_params,)
    assert distribution.jac_cdf(time()).shape == (distribution.nb_params,)
    assert distribution.jac_pdf(time()).shape == (distribution.nb_params,)

    assert distribution.dhf(time(n,)).shape == (n,)
    assert distribution.jac_sf(time(n,)).shape == (distribution.nb_params, n)
    assert distribution.jac_hf(time(n,)).shape == (distribution.nb_params, n)
    assert distribution.jac_chf(time(n,)).shape == (distribution.nb_params, n)
    assert distribution.jac_cdf(time(n,)).shape == (distribution.nb_params, n)
    assert distribution.jac_pdf(time(n,)).shape == (distribution.nb_params, n)

    assert distribution.dhf(time(m, 1)).shape == (m, 1)
    assert distribution.jac_sf(time(m, 1)).shape == (distribution.nb_params, m, 1)
    assert distribution.jac_hf(time(m, 1)).shape == (distribution.nb_params, m, 1)
    assert distribution.jac_chf(time(m, 1)).shape == (distribution.nb_params, m, 1)
    assert distribution.jac_cdf(time(m, 1)).shape == (distribution.nb_params, m, 1)
    assert distribution.jac_pdf(time(m, 1)).shape == (distribution.nb_params, m, 1)

    assert distribution.dhf(time(m, n)).shape == (m, n)
    assert distribution.jac_sf(time(m, n)).shape == (distribution.nb_params, m, n)
    assert distribution.jac_hf(time(m, n)).shape == (distribution.nb_params, m, n)
    assert distribution.jac_chf(time(m, n)).shape == (distribution.nb_params, m, n)
    assert distribution.jac_cdf(time(m, n)).shape == (distribution.nb_params, m, n)
    assert distribution.jac_pdf(time(m, n)).shape == (distribution.nb_params, m, n)


def test_ls_integrate(distribution, a, b):
    m = 2
    n = 3

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(), b())
    assert integration.shape == ()
    assert integration == approx(distribution.cdf(b()) - distribution.cdf(a()))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, 0.0, np.inf, deg=100)
    assert integration == approx(distribution.mean(), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(), b(n))
    assert integration.shape == (n,)
    assert integration == approx(distribution.cdf(b(n)) - distribution.cdf(a()))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, 0.0, np.full((n,), np.inf))
    assert integration == approx(np.full((n,), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(n), b())
    assert integration.shape == (n,)
    assert integration == approx(distribution.cdf(b()) - distribution.cdf(a(n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros(n), np.inf)
    assert integration == approx(np.full((n,), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(n), b(n))
    assert integration.shape == (n,)
    assert integration == approx(distribution.cdf(b(n)) - distribution.cdf(a(n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros(n), np.full((n,), np.inf))
    assert integration == approx(np.full((n,), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a()))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, 0.0, np.full((m, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(m, n), b())
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b()) - distribution.cdf(a(m, n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros((m, n)), np.inf)
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(m, 1), b(1, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(1, n)) - distribution.cdf(a(m, 1)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros((m,1)), np.full((1, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(1, n), b(m, 1))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, 1)) - distribution.cdf(a(1, n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros((m,n)), np.full((m, 1), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(m, 1), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a(1, n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros((m,1)), np.full((m, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(1, n), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a(1, n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros((m,n)), np.full((m, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(m, n), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a(m, n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros((m, n)), np.full((m, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)


def test_fit(distribution, power_transformer_data):
    expected_params = distribution.params.copy()
    distribution = distribution.fit(
        power_transformer_data[0, :],
        event=power_transformer_data[1, :] == 1,
        entry=power_transformer_data[2, :],
    )
    assert distribution.params == pytest.approx(expected_params, rel=1e-3)