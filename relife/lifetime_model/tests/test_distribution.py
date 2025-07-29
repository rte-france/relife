import numpy as np
import pytest
from pytest import approx


def test_rvs(distribution, rvs_size, expected_out_shape):
    assert distribution.rvs(size=rvs_size).shape == expected_out_shape(size=rvs_size)
    assert all(
        arr.shape == expected_out_shape(size=rvs_size) for arr in distribution.rvs(size=rvs_size, return_event=True)
    )
    assert all(
        arr.shape == expected_out_shape(size=rvs_size) for arr in distribution.rvs(size=rvs_size, return_entry=True)
    )
    assert all(
        arr.shape == expected_out_shape(size=rvs_size)
        for arr in distribution.rvs(size=rvs_size, return_event=True, return_entry=True)
    )


def test_sf(distribution, time, expected_out_shape):
    assert distribution.sf(time).shape == expected_out_shape(time=time)
    assert distribution.sf(np.full(expected_out_shape(time=time), distribution.median())) == approx(
        np.full(expected_out_shape(time=time), 0.5), rel=1e-3
    )


def test_hf(distribution, time, expected_out_shape):
    assert distribution.hf(time).shape == expected_out_shape(time=time)


def test_chf(distribution, time, expected_out_shape):
    assert distribution.chf(time).shape == expected_out_shape(time=time)


def test_cdf(distribution, time, expected_out_shape):
    assert distribution.cdf(time).shape == expected_out_shape(time=time)


def test_pdf(distribution, time, expected_out_shape):
    assert distribution.pdf(time).shape == expected_out_shape(time=time)


def test_ppf(distribution, probability, expected_out_shape):
    assert distribution.ppf(probability).shape == expected_out_shape(probability=probability)


def test_ichf(distribution, probability, expected_out_shape):
    assert distribution.ichf(probability).shape == expected_out_shape(probability=probability)


def test_isf(distribution, probability, expected_out_shape):
    assert distribution.isf(probability).shape == expected_out_shape(probability=probability)
    assert distribution.isf(np.full(expected_out_shape(probability=probability), 0.5)) == approx(
        np.full(expected_out_shape(probability=probability), distribution.median())
    )


def test_moment(distribution, expected_out_shape):
    assert distribution.moment(1).shape == expected_out_shape()
    assert distribution.moment(2).shape == expected_out_shape()


def test_mean(distribution, expected_out_shape):
    assert distribution.mean().shape == expected_out_shape()


def test_var(distribution, expected_out_shape):
    assert distribution.var().shape == expected_out_shape()


def test_median(distribution, expected_out_shape):
    assert distribution.median().shape == expected_out_shape()


def test_dhf(distribution, time, expected_out_shape):
    assert distribution.dhf(time).shape == expected_out_shape(time=time)


def test_jac_sf(distribution, time, expected_out_shape):
    assert distribution.jac_sf(time, asarray=True).shape == (distribution.nb_params,) + expected_out_shape(time=time)


def test_jac_hf(distribution, time, expected_out_shape):
    assert distribution.jac_hf(time, asarray=True).shape == (distribution.nb_params,) + expected_out_shape(time=time)


def test_jac_chf(distribution, time, expected_out_shape):
    assert distribution.jac_chf(time, asarray=True).shape == (distribution.nb_params,) + expected_out_shape(time=time)


def test_jac_cdf(distribution, time, expected_out_shape):
    assert distribution.jac_cdf(time, asarray=True).shape == (distribution.nb_params,) + expected_out_shape(time=time)


def test_jac_pdf(distribution, time, expected_out_shape):
    assert distribution.jac_pdf(time, asarray=True).shape == (distribution.nb_params,) + expected_out_shape(time=time)


def test_ls_integrate(distribution, integration_bound_a, integration_bound_b, expected_out_shape):
    # integral_a^b dF(x)
    shape = expected_out_shape(integration_bound_a=integration_bound_a, integration_bound_b=integration_bound_b)
    integration = distribution.ls_integrate(np.ones_like, integration_bound_a, integration_bound_b, deg=100)
    assert integration.shape == shape
    assert integration == approx(distribution.cdf(integration_bound_b) - distribution.cdf(integration_bound_a))
    # integral_0^inf x*dF(x)
    integration = distribution.ls_integrate(
        lambda x: x, np.zeros_like(integration_bound_a), np.full_like(integration_bound_b, np.inf), deg=100
    )
    assert integration == approx(np.full(shape, distribution.mean()), rel=1e-3)


def test_fit(distribution, power_transformer_data):
    expected_params = distribution.params.copy()
    distribution = distribution.fit(
        power_transformer_data["time"],
        event=power_transformer_data["event"],
        entry=power_transformer_data["entry"],
    )
    assert distribution.params == pytest.approx(expected_params, rel=1e-3)


class TestEquilibriumDistribution:

    # def test_args_names(self, equilibrium_distribution):
    #     assert equilibrium_distribution.args_names == ()
    #
    # def test_rvs(self, equilibrium_distribution):
    #     m, n = 3, 10
    #     assert equilibrium_distribution.rvs(seed=21).shape == ()
    #     assert equilibrium_distribution.rvs(size=(n,), seed=21).shape == (n,)
    #     assert equilibrium_distribution.rvs(size=(m, 1), seed=21).shape == (m, 1)
    #     assert equilibrium_distribution.rvs(size=(m, n), seed=21).shape == (m, n)

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
