from typing import TypeAlias

import numpy as np
import pytest
from optype.numpy import Array1D, ArrayND
from pytest import approx

from relife.lifetime_models import (
    Exponential,
    Gamma,
    Gompertz,
    LifetimeLikelihood,
    LogLogistic,
    Weibull,
)

_Distrib: TypeAlias = Exponential | Weibull | Gompertz | LogLogistic | Gamma
ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


class TestBroadcasting:
    def test_rvs(self, distribution: _Distrib, rvs_size: int | tuple[int, ...]):
        expected_shape = (rvs_size,) if isinstance(rvs_size, int) else rvs_size
        assert distribution.rvs(rvs_size).shape == expected_shape

    def test_sf(self, distribution: _Distrib, time: ArrayND[np.float64]):
        assert distribution.sf(time).shape == np.broadcast(time).shape

    def test_hf(self, distribution: _Distrib, time: ArrayND[np.float64]):
        assert distribution.hf(time).shape == np.broadcast(time).shape

    def test_chf(self, distribution: _Distrib, time: ArrayND[np.float64]):
        assert distribution.chf(time).shape == np.broadcast(time).shape

    def test_cdf(self, distribution: _Distrib, time: ArrayND[np.float64]):
        assert distribution.cdf(time).shape == np.broadcast(time).shape

    def test_pdf(self, distribution: _Distrib, time: ArrayND[np.float64]):
        assert distribution.pdf(time).shape == np.broadcast(time).shape

    def test_ppf(self, distribution: _Distrib, probability: ArrayND[np.float64]):
        assert distribution.ppf(probability).shape == np.broadcast(probability).shape

    def test_ichf(self, distribution: _Distrib, probability: ArrayND[np.float64]):
        assert distribution.ichf(probability).shape == np.broadcast(probability).shape

    def test_isf(self, distribution: _Distrib, probability: ArrayND[np.float64]):
        assert distribution.isf(probability).shape == np.broadcast(probability).shape

    def test_moment(self, distribution: _Distrib):
        assert distribution.moment(1).shape == ()
        assert distribution.moment(2).shape == ()

    def test_mean(self, distribution: _Distrib):
        assert distribution.mean().shape == ()

    def test_var(self, distribution: _Distrib):
        assert distribution.var().shape == ()

    def test_median(self, distribution: _Distrib):
        assert distribution.median().shape == ()

    def test_dhf(self, distribution: _Distrib, time: ArrayND[np.float64]):
        assert distribution.dhf(time).shape == time.shape

    def test_jac_sf(self, distribution: _Distrib, time: ArrayND[np.float64]):
        nb_params = distribution.get_params().size
        assert distribution.jac_sf(time).shape == (nb_params,) + time.shape

    def test_jac_hf(self, distribution: _Distrib, time: ArrayND[np.float64]):
        nb_params = distribution.get_params().size
        assert distribution.jac_hf(time).shape == (nb_params,) + time.shape

    def test_jac_chf(self, distribution: _Distrib, time: ArrayND[np.float64]):
        nb_params = distribution.get_params().size
        assert distribution.jac_chf(time).shape == (nb_params,) + time.shape

    def test_jac_cdf(self, distribution: _Distrib, time: ArrayND[np.float64]):
        nb_params = distribution.get_params().size
        assert distribution.jac_cdf(time).shape == (nb_params,) + time.shape

    def test_jac_pdf(self, distribution: _Distrib, time: ArrayND[np.float64]):
        nb_params = distribution.get_params().size
        assert distribution.jac_pdf(time).shape == (nb_params,) + time.shape

    def test_ls_integrate(
        self,
        distribution: _Distrib,
        integration_bound_a: ArrayND[np.float64],
        integration_bound_b: ArrayND[np.float64],
    ):
        expected_shape = np.broadcast_shapes(
            integration_bound_a.shape, integration_bound_b.shape
        )
        integration = distribution.ls_integrate(
            np.ones_like, integration_bound_a, integration_bound_b
        )
        assert integration.shape == expected_shape


def test_sf_values(distribution: _Distrib, time: ArrayND[np.float64]):
    assert distribution.sf(np.full(time.shape, distribution.median())) == approx(
        np.full(time.shape, 0.5), rel=1e-3
    )


def test_isf_values(distribution: _Distrib, probability: ArrayND[np.float64]):
    assert distribution.isf(np.full(probability.shape, 0.5)) == approx(
        np.full(probability.shape, distribution.median())
    )


def test_ls_integrate_values(
    distribution: _Distrib,
    integration_bound_a: ArrayND[np.float64],
    integration_bound_b: ArrayND[np.float64],
):
    integration = distribution.ls_integrate(
        np.ones_like, integration_bound_a, integration_bound_b, deg=100
    )
    assert integration == approx(
        distribution.cdf(integration_bound_b) - distribution.cdf(integration_bound_a)
    )
    integration = distribution.ls_integrate(
        lambda x: x,
        np.zeros_like(integration_bound_a),
        np.full_like(integration_bound_b, np.inf),
        deg=100,
    )
    assert integration == approx(
        np.full(integration.shape, distribution.mean()), rel=1e-3
    )


def test_fit(distribution: _Distrib, power_transformer_data: Array1D[np.void]):
    expected_params = distribution.get_params().copy()
    distribution = distribution.fit(
        power_transformer_data["time"],
        event=power_transformer_data["event"],
        entry=power_transformer_data["entry"],
    )
    assert distribution.get_params() == pytest.approx(expected_params, rel=1e-3)


def test_negative_log(distribution_likelihood: LifetimeLikelihood[_Distrib]):
    params = distribution_likelihood.model.get_params().copy()
    assert isinstance(distribution_likelihood.negative_log(params), float)


def test_jac_negative_log(distribution_likelihood: LifetimeLikelihood[_Distrib]):
    params = distribution_likelihood.model.get_params().copy()
    assert distribution_likelihood.jac_negative_log(params).shape == (params.size,)


class TestEquilibriumDistribution:
    # def test_args_names(self, equilibrium_distribution:_Distrib):
    #     assert equilibrium_distribution.args_names == ()
    #
    # def test_rvs(self, equilibrium_distribution:_Distrib):
    #     m, n = 3, 10
    #     assert equilibrium_distribution.rvs(seed=21).shape == ()
    #     assert equilibrium_distribution.rvs(size=(n,), seed=21).shape == (n,)
    #     assert equilibrium_distribution.rvs(size=(m, 1), seed=21).shape == (m, 1)
    #     assert equilibrium_distribution.rvs(size=(m, n), seed=21).shape == (m, n)

    @pytest.mark.xfail
    def test_moment(self, equilibrium_distribution: _Distrib):
        assert equilibrium_distribution.moment(1).shape == ()
        assert equilibrium_distribution.moment(2).shape == ()

    @pytest.mark.xfail
    def test_mean(self, equilibrium_distribution: _Distrib):
        assert equilibrium_distribution.mean().shape == ()

    @pytest.mark.xfail
    def test_var(self, equilibrium_distribution: _Distrib):
        assert equilibrium_distribution.var().shape == ()

    @pytest.mark.xfail
    def test_median(self, equilibrium_distribution: _Distrib):
        assert equilibrium_distribution.median().shape == ()
