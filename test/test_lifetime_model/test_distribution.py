import pytest
from pytest import approx
import numpy as np

def test_args_names(distribution, equilibrium_distribution):
    assert distribution.args_names == ()

def test_args_names_equilibrium_distribution(equilibrium_distribution):
    assert equilibrium_distribution.args_names == ()

def test_rvs(distribution):
    m, n = 3, 10
    assert distribution.rvs(seed=21).shape == ()
    assert distribution.rvs((n,), seed=21).shape == (n,)
    assert distribution.rvs((m, 1), seed=21).shape == (m, 1)
    assert distribution.rvs((m, n), seed=21).shape == (m, n)

def test_rvs_equilibrium_distribution(equilibrium_distribution):
    m, n = 3, 10
    assert equilibrium_distribution.rvs(seed=21).shape == ()
    assert equilibrium_distribution.rvs(shape=(n,), seed=21).shape == (n,)
    assert equilibrium_distribution.rvs(shape=(m, 1), seed=21).shape == (m, 1)
    assert equilibrium_distribution.rvs(shape=(m, n), seed=21).shape == (m, n)

def test_sf(distribution, time):
    assert distribution.sf(time).shape == time.shape
    assert distribution.sf(np.full(time.shape, distribution.median())) == approx(np.full(time.shape, 0.5), rel=1e-3)

def test_hf(distribution, time):
    assert distribution.hf(time).shape == time.shape

def test_chf(distribution, time):
    assert distribution.chf(time).shape == time.shape

def test_cdf(distribution, time):
    assert distribution.cdf(time).shape == time.shape

def test_pdf(distribution, time):
    assert distribution.pdf(time).shape == time.shape

def test_ppf(distribution, probability):
    assert distribution.ppf(probability).shape == probability.shape

def test_ichf(distribution, probability):
    assert distribution.ichf(probability).shape == probability.shape

def test_isf(distribution, probability):
    assert distribution.isf(probability).shape == probability.shape
    assert distribution.isf(np.full(probability.shape, 0.5)) == approx(np.full(probability.shape, distribution.median()))

def test_moment(distribution, time):
    assert distribution.moment(1).shape == ()
    assert distribution.moment(2).shape == ()
    assert distribution.mean().shape == ()
    assert distribution.var().shape == ()
    assert distribution.median().shape == ()

@pytest.mark.xfail
def test_moment_equilibrium_distribution(equilibrium_distribution, time):
    assert equilibrium_distribution.moment(1).shape == ()
    assert equilibrium_distribution.moment(2).shape == ()
    assert equilibrium_distribution.mean().shape == ()
    assert equilibrium_distribution.var().shape == ()
    assert equilibrium_distribution.median().shape == ()

def test_dhf(distribution, time):
    assert distribution.dhf(time).shape == time.shape

def test_jac_sf(distribution, time):
    assert distribution.jac_sf(time, asarray=True).shape == (distribution.nb_params,) + time.shape

def test_jac_hf(distribution, time):
    assert distribution.jac_hf(time, asarray=True).shape == (distribution.nb_params,) + time.shape

def test_jac_chf(distribution, time):
    assert distribution.jac_chf(time, asarray=True).shape == (distribution.nb_params,) + time.shape

def test_jac_cdf(distribution, time):
    assert distribution.jac_cdf(time, asarray=True).shape == (distribution.nb_params,) + time.shape

def test_jac_pdf(distribution, time):
    assert distribution.jac_pdf(time, asarray=True).shape == (distribution.nb_params,) + time.shape


def test_ls_integrate(distribution, a, b):
    m = 2
    n = 3

    # integral_a^b dF(x)
    integration = distribution.ls_integrate(np.ones_like, a(), b(), deg=100)
    assert integration.shape == ()
    assert integration == approx(distribution.cdf(b()) - distribution.cdf(a()))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, 0.0, np.inf, deg=100)
    assert integration == approx(distribution.mean(), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(), b(n), deg=100)
    assert integration.shape == (n,)
    assert integration == approx(distribution.cdf(b(n)) - distribution.cdf(a()))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, 0.0, np.full((n,), np.inf), deg=100)
    assert integration == approx(np.full((n,), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(n), b(), deg=100)
    assert integration.shape == (n,)
    assert integration == approx(distribution.cdf(b()) - distribution.cdf(a(n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros(n), np.inf, deg=100)
    assert integration == approx(np.full((n,), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(n), b(n), deg=100)
    assert integration.shape == (n,)
    assert integration == approx(distribution.cdf(b(n)) - distribution.cdf(a(n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros(n), np.full((n,), np.inf), deg=100)
    assert integration == approx(np.full((n,), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(), b(m, n), deg=100)
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a()))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, 0.0, np.full((m, n), np.inf), deg=100)
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(m, n), b(), deg=100)
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b()) - distribution.cdf(a(m, n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros((m, n)), np.inf, deg=100)
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(m, 1), b(1, n), deg=100)
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(1, n)) - distribution.cdf(a(m, 1)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros((m,1)), np.full((1, n), np.inf), deg=100)
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(1, n), b(m, 1), deg=100)
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, 1)) - distribution.cdf(a(1, n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros((m,n)), np.full((m, 1), np.inf), deg=100)
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(m, 1), b(m, n), deg=100)
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a(1, n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros((m,1)), np.full((m, n), np.inf), deg=100)
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(1, n), b(m, n), deg=100)
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a(1, n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros((m,n)), np.full((m, n), np.inf), deg=100)
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = distribution.ls_integrate( np.ones_like, a(m, n), b(m, n), deg=100)
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a(m, n)))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros((m, n)), np.full((m, n), np.inf), deg=100)
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

# @pytest.mark.xfail
def test_fit(distribution, power_transformer_data):
    expected_params = distribution.params.copy()
    distribution = distribution.fit(
        power_transformer_data[0, :],
        event=power_transformer_data[1, :] == 1,
        entry=power_transformer_data[2, :],
    )
    assert distribution.params == pytest.approx(expected_params, rel=1e-3)