import pytest
from pytest import approx
import numpy as np


class TestDistribution:
    def test_args_names(self, distribution):
        assert distribution.args_names == ()

    def test_rvs(self, distribution):
        m, n = 3, 10
        assert distribution.rvs(seed=21).shape == ()
        assert distribution.rvs(n, seed=21).shape == (n,)
        assert distribution.rvs((n,), seed=21).shape == (n,)
        assert distribution.rvs((m, 1), seed=21).shape == (m, 1)
        assert distribution.rvs((m, n), seed=21).shape == (m, n)

    def test_sf(self, distribution, time):
        assert distribution.sf(time).shape == time.shape
        assert distribution.sf(np.full(time.shape, distribution.median())) == approx(np.full(time.shape, 0.5), rel=1e-3)

    def test_hf(self, distribution, time):
        assert distribution.hf(time).shape == time.shape

    def test_chf(self, distribution, time):
        assert distribution.chf(time).shape == time.shape

    def test_cdf(self, distribution, time):
        assert distribution.cdf(time).shape == time.shape

    def test_pdf(self, distribution, time):
        assert distribution.pdf(time).shape == time.shape

    def test_ppf(self, distribution, probability):
        assert distribution.ppf(probability).shape == probability.shape

    def test_ichf(self, distribution, probability):
        assert distribution.ichf(probability).shape == probability.shape

    def test_isf(self, distribution, probability):
        assert distribution.isf(probability).shape == probability.shape
        assert distribution.isf(np.full(probability.shape, 0.5)) == approx(np.full(probability.shape, distribution.median()))

    def test_moment(self, distribution):
        assert distribution.moment(1).shape == ()
        assert distribution.moment(2).shape == ()

    def test_mean(self, distribution):
        assert distribution.mean().shape == ()

    def test_var(self, distribution):
        assert distribution.var().shape == ()

    def test_median(self, distribution):
        assert distribution.median().shape == ()

    def test_dhf(self, distribution, time):
        assert distribution.dhf(time).shape == time.shape

    def test_jac_sf(self, distribution, time):
        assert distribution.jac_sf(time, asarray=True).shape == (distribution.nb_params,) + time.shape

    def test_jac_hf(self, distribution, time):
        assert distribution.jac_hf(time, asarray=True).shape == (distribution.nb_params,) + time.shape

    def test_jac_chf(self, distribution, time):
        assert distribution.jac_chf(time, asarray=True).shape == (distribution.nb_params,) + time.shape

    def test_jac_cdf(self, distribution, time):
        assert distribution.jac_cdf(time, asarray=True).shape == (distribution.nb_params,) + time.shape

    def test_jac_pdf(self, distribution, time):
        assert distribution.jac_pdf(time, asarray=True).shape == (distribution.nb_params,) + time.shape

    def test_ls_integrate(self, distribution, integration_bound_a, integration_bound_b):
        # integral_a^b dF(x)
        shape = np.broadcast_shapes(integration_bound_a.shape, integration_bound_b.shape)
        integration = distribution.ls_integrate(np.ones_like, integration_bound_a, integration_bound_b, deg=100)
        assert integration.shape == shape
        assert integration == approx(distribution.cdf(integration_bound_b) - distribution.cdf(integration_bound_a))
        # integral_0^inf x*dF(x)
        integration = distribution.ls_integrate(lambda x: x, np.zeros_like(integration_bound_a), np.full_like(integration_bound_b, np.inf), deg=100)
        assert integration == approx(np.full(shape, distribution.mean()), rel=1e-3)

    def test_fit(self, distribution, power_transformer_data):
        expected_params = distribution.params.copy()
        distribution = distribution.fit(
            power_transformer_data[0, :],
            event=power_transformer_data[1, :] == 1,
            entry=power_transformer_data[2, :],
        )
        assert distribution.params == pytest.approx(expected_params, rel=1e-3)

class TestEquilibriumDistribution:

    def test_args_names(self, equilibrium_distribution):
        assert equilibrium_distribution.args_names == ()

    def test_rvs(self, equilibrium_distribution):
        m, n = 3, 10
        assert equilibrium_distribution.rvs(seed=21).shape == ()
        assert equilibrium_distribution.rvs(size=(n,), seed=21).shape == (n,)
        assert equilibrium_distribution.rvs(size=(m, 1), seed=21).shape == (m, 1)
        assert equilibrium_distribution.rvs(size=(m, n), seed=21).shape == (m, n)

    @pytest.mark.xfail
    def test_moment(self, equilibrium_distribution):
        assert equilibrium_distribution.moment(1).shape == ()
        assert equilibrium_distribution.moment(2).shape == ()

    @pytest.mark.xfail
    def test_mean(self, equilibrium_distribution):
        assert equilibrium_distribution.mean().shape == ()

    @pytest.mark.xfail
    def test_var(self, equilibrium_distribution):
        assert equilibrium_distribution.var().shape == ()

    @pytest.mark.xfail
    def test_median(self, equilibrium_distribution):
        assert equilibrium_distribution.median().shape == ()



