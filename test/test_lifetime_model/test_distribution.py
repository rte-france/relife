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
    assert distribution.rvs(n, seed=21).shape == (n,)
    assert distribution.rvs((n,), seed=21).shape == (n,)
    assert distribution.rvs((m, 1), seed=21).shape == (m, 1)
    assert distribution.rvs((m, n), seed=21).shape == (m, n)

def test_rvs_equilibrium_distribution(equilibrium_distribution):
    m, n = 3, 10
    assert equilibrium_distribution.rvs(seed=21).shape == ()
    assert equilibrium_distribution.rvs(size=(n,), seed=21).shape == (n,)
    assert equilibrium_distribution.rvs(size=(m, 1), seed=21).shape == (m, 1)
    assert equilibrium_distribution.rvs(size=(m, n), seed=21).shape == (m, n)

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


def test_ls_integrate(distribution, integration_bounds):
    # integral_a^b dF(x)
    a, b = integration_bounds
    shape = np.broadcast_shapes(a.shape, b.shape)
    integration = distribution.ls_integrate(np.ones_like, a, b ,deg=100)
    assert integration.shape == shape
    assert integration == approx(distribution.cdf(b) - distribution.cdf(a))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate( lambda x: x, np.zeros_like(a), np.full_like(b, np.inf), deg=100)
    assert integration == approx(np.full(shape, distribution.mean()), rel=1e-3)

# @pytest.mark.xfail
def test_fit(distribution, power_transformer_data):
    expected_params = distribution.params.copy()
    distribution = distribution.fit(
        power_transformer_data[0, :],
        event=power_transformer_data[1, :] == 1,
        entry=power_transformer_data[2, :],
    )
    assert distribution.params == pytest.approx(expected_params, rel=1e-3)