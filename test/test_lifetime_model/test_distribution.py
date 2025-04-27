import pytest
from pytest import approx
import numpy as np

from relife.lifetime_model import EquilibriumDistribution

def test_args_names(distribution):
    assert distribution.args_names == ()
    assert EquilibriumDistribution(distribution).args_names == ()

def test_rvs(distribution):
    m, n = 3, 10
    assert isinstance(distribution.rvs(seed=21), float)
    assert distribution.rvs((n,), seed=21).shape == (n,)
    assert distribution.rvs((m, 1), seed=21).shape == (m, 1)
    assert distribution.rvs((m, n), seed=21).shape == (m, n)

def test_probility_functions(distribution, time, probability):
    m, n = 3, 10

    assert isinstance(distribution.sf(time()), float)
    assert distribution.sf(distribution.median()) == approx(0.5, rel=1e-3)
    assert isinstance(distribution.hf(time()), float)
    assert isinstance(distribution.chf(time()), float)
    assert isinstance(distribution.cdf(time()), float)
    assert isinstance(distribution.pdf(time()), float)
    assert isinstance(distribution.ppf(probability()), float)
    assert isinstance(distribution.ichf(probability()), float)
    assert isinstance(distribution.isf(probability()), float)
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
    assert isinstance(distribution.moment(1), float)
    assert isinstance(distribution.moment(2), float)
    assert isinstance(distribution.mean(), float)
    assert isinstance(distribution.var(), float)
    assert isinstance(distribution.median(), float)


def test_derivative(distribution, time):
    m, n = 3, 10

    assert isinstance(distribution.dhf(time()), float)
    jac_sf = distribution.jac_sf(time())
    jac_hf = distribution.jac_hf(time())
    jac_chf = distribution.jac_chf(time())
    jac_cdf = distribution.jac_cdf(time())
    jac_pdf = distribution.jac_pdf(time())
    if distribution.nb_params > 1:
        assert len(jac_sf) == distribution.nb_params
        assert len(jac_hf) == distribution.nb_params
        assert len(jac_chf) == distribution.nb_params
        assert len(jac_cdf) == distribution.nb_params
        assert len(jac_pdf) == distribution.nb_params
        assert all(jac.shape == () for jac in jac_sf)
        assert all(jac.shape == () for jac in jac_hf)
        assert all(jac.shape == () for jac in jac_chf)
        assert all(jac.shape == () for jac in jac_cdf)
        assert all(jac.shape == () for jac in jac_pdf)
    else:
        assert isinstance(jac_sf, float)
        assert isinstance(jac_hf, float)
        assert isinstance(jac_chf, float)
        assert isinstance(jac_cdf, float)
        assert isinstance(jac_pdf, float)

    assert distribution.dhf(time(n)).shape == (n,)
    jac_sf = distribution.jac_sf(time(n))
    jac_hf = distribution.jac_hf(time(n))
    jac_chf = distribution.jac_chf(time(n))
    jac_cdf = distribution.jac_cdf(time(n))
    jac_pdf = distribution.jac_pdf(time(n))
    if distribution.nb_params > 1:
        assert len(jac_sf) == distribution.nb_params
        assert len(jac_hf) == distribution.nb_params
        assert len(jac_chf) == distribution.nb_params
        assert len(jac_cdf) == distribution.nb_params
        assert len(jac_pdf) == distribution.nb_params
        assert all(jac.shape == (n,) for jac in jac_sf)
        assert all(jac.shape == (n,) for jac in jac_hf)
        assert all(jac.shape == (n,) for jac in jac_chf)
        assert all(jac.shape == (n,) for jac in jac_cdf)
        assert all(jac.shape == (n,) for jac in jac_pdf)
    else:
        assert jac_sf.shape == (n,)
        assert jac_hf.shape == (n,)
        assert jac_chf.shape == (n,)
        assert jac_cdf.shape == (n,)
        assert jac_pdf.shape == (n,)

    assert distribution.dhf(time(m, 1)).shape == (m, 1)
    jac_sf = distribution.jac_sf(time(m,1))
    jac_hf = distribution.jac_hf(time(m,1))
    jac_chf = distribution.jac_chf(time(m,1))
    jac_cdf = distribution.jac_cdf(time(m,1))
    jac_pdf = distribution.jac_pdf(time(m,1))
    if distribution.nb_params > 1:
        assert len(jac_sf) == distribution.nb_params
        assert len(jac_hf) == distribution.nb_params
        assert len(jac_chf) == distribution.nb_params
        assert len(jac_cdf) == distribution.nb_params
        assert len(jac_pdf) == distribution.nb_params
        assert all(jac.shape == (m,1) for jac in jac_sf)
        assert all(jac.shape == (m,1) for jac in jac_hf)
        assert all(jac.shape == (m,1) for jac in jac_chf)
        assert all(jac.shape == (m,1) for jac in jac_cdf)
        assert all(jac.shape == (m,1) for jac in jac_pdf)
    else:
        assert jac_sf.shape == (m,1)
        assert jac_hf.shape == (m,1)
        assert jac_chf.shape == (m,1)
        assert jac_cdf.shape == (m,1)
        assert jac_pdf.shape == (m,1)

    assert distribution.dhf(time(m, n)).shape == (m, n)
    jac_sf = distribution.jac_sf(time(m,n))
    jac_hf = distribution.jac_hf(time(m,n))
    jac_chf = distribution.jac_chf(time(m,n))
    jac_cdf = distribution.jac_cdf(time(m,n))
    jac_pdf = distribution.jac_pdf(time(m,n))
    if distribution.nb_params > 1:
        assert len(jac_sf) == distribution.nb_params
        assert len(jac_hf) == distribution.nb_params
        assert len(jac_chf) == distribution.nb_params
        assert len(jac_cdf) == distribution.nb_params
        assert len(jac_pdf) == distribution.nb_params
        assert all(jac.shape == (m,n) for jac in jac_sf)
        assert all(jac.shape == (m,n) for jac in jac_hf)
        assert all(jac.shape == (m,n) for jac in jac_chf)
        assert all(jac.shape == (m,n) for jac in jac_cdf)
        assert all(jac.shape == (m,n) for jac in jac_pdf)
    else:
        assert jac_sf.shape == (m,n)
        assert jac_hf.shape == (m,n)
        assert jac_chf.shape == (m,n)
        assert jac_cdf.shape == (m,n)
        assert jac_pdf.shape == (m,n)


def test_ls_integrate(distribution, a, b):
    m = 2
    n = 3

    # integral_a^b dF(x)
    integration = distribution.ls_integrate(np.ones_like, a(), b())
    assert isinstance(integration, float)
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