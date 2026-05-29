from typing import TypeAlias

import numpy as np
import pytest
from numpy.testing import assert_allclose
from optype.numpy import Array1D, ArrayND

from relife.lifetime_models import (
    LifetimeLikelihood,
)
from relife.lifetime_models._distributions import LifetimeDistribution
from relife.utils import to_numpy_float64

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


class TestBroadcasting:
    def test_rvs(
        self, distribution: LifetimeDistribution, rvs_size: int | tuple[int, ...]
    ):
        expected_shape = (rvs_size,) if isinstance(rvs_size, int) else rvs_size
        assert distribution.rvs(rvs_size).shape == expected_shape

    def test_sf(self, distribution: LifetimeDistribution, time: ArrayND[np.float64]):
        assert distribution.sf(time).shape == np.broadcast(time).shape

    def test_hf(self, distribution: LifetimeDistribution, time: ArrayND[np.float64]):
        assert distribution.hf(time).shape == np.broadcast(time).shape

    def test_chf(self, distribution: LifetimeDistribution, time: ArrayND[np.float64]):
        assert distribution.chf(time).shape == np.broadcast(time).shape

    def test_cdf(self, distribution: LifetimeDistribution, time: ArrayND[np.float64]):
        assert distribution.cdf(time).shape == np.broadcast(time).shape

    def test_pdf(self, distribution: LifetimeDistribution, time: ArrayND[np.float64]):
        assert distribution.pdf(time).shape == np.broadcast(time).shape

    def test_ppf(
        self, distribution: LifetimeDistribution, probability: ArrayND[np.float64]
    ):
        assert distribution.ppf(probability).shape == np.broadcast(probability).shape

    def test_ichf(
        self, distribution: LifetimeDistribution, probability: ArrayND[np.float64]
    ):
        assert distribution.ichf(probability).shape == np.broadcast(probability).shape

    def test_isf(
        self, distribution: LifetimeDistribution, probability: ArrayND[np.float64]
    ):
        assert distribution.isf(probability).shape == np.broadcast(probability).shape

    def test_moment(self, distribution: LifetimeDistribution):
        assert distribution.moment(1).shape == ()
        assert distribution.moment(2).shape == ()

    def test_mean(self, distribution: LifetimeDistribution):
        assert distribution.mean().shape == ()

    def test_var(self, distribution: LifetimeDistribution):
        assert distribution.var().shape == ()

    def test_median(self, distribution: LifetimeDistribution):
        assert distribution.median().shape == ()

    def test_dhf(self, distribution: LifetimeDistribution, time: ArrayND[np.float64]):
        assert distribution.dhf(time).shape == time.shape

    def test_jac_sf(
        self, distribution: LifetimeDistribution, time: ArrayND[np.float64]
    ):
        nb_params = distribution.get_params().size
        assert distribution.jac_sf(time).shape == (nb_params,) + time.shape

    def test_jac_hf(
        self, distribution: LifetimeDistribution, time: ArrayND[np.float64]
    ):
        nb_params = distribution.get_params().size
        assert distribution.jac_hf(time).shape == (nb_params,) + time.shape

    def test_jac_chf(
        self, distribution: LifetimeDistribution, time: ArrayND[np.float64]
    ):
        nb_params = distribution.get_params().size
        assert distribution.jac_chf(time).shape == (nb_params,) + time.shape

    def test_jac_cdf(
        self, distribution: LifetimeDistribution, time: ArrayND[np.float64]
    ):
        nb_params = distribution.get_params().size
        assert distribution.jac_cdf(time).shape == (nb_params,) + time.shape

    def test_jac_pdf(
        self, distribution: LifetimeDistribution, time: ArrayND[np.float64]
    ):
        nb_params = distribution.get_params().size
        assert distribution.jac_pdf(time).shape == (nb_params,) + time.shape

    def test_ls_integrate(
        self,
        distribution: LifetimeDistribution,
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


def test_sf_values(distribution: LifetimeDistribution, time: ArrayND[np.float64]):
    assert_allclose(
        distribution.sf(np.full(time.shape, distribution.median())),
        np.full(time.shape, 0.5),
        rtol=1e-3,
    )


def test_isf_values(
    distribution: LifetimeDistribution, probability: ArrayND[np.float64]
):
    assert_allclose(
        distribution.isf(np.full(probability.shape, 0.5)),
        np.full(probability.shape, distribution.median()),
    )


def test_ls_integrate_values(
    distribution: LifetimeDistribution,
    integration_bound_a: ArrayND[np.float64],
    integration_bound_b: ArrayND[np.float64],
):
    integration = distribution.ls_integrate(
        np.ones_like, integration_bound_a, integration_bound_b, deg=100
    )
    assert_allclose(
        integration,
        distribution.cdf(integration_bound_b) - distribution.cdf(integration_bound_a),
    )

    def func(x: ST | NumpyST | ArrayND[NumpyST]) -> np.float64 | ArrayND[np.float64]:
        return to_numpy_float64(x)

    integration = distribution.ls_integrate(
        func,
        np.zeros_like(integration_bound_a),
        np.full_like(integration_bound_b, np.inf),
        deg=100,
    )
    assert_allclose(
        integration, np.full(integration.shape, distribution.mean()), rtol=1e-3
    )


def test_fit(
    distribution: LifetimeDistribution, power_transformer_data: Array1D[np.void]
):
    expected_params = distribution.get_params().copy()
    distribution = distribution.fit(
        power_transformer_data["time"],
        event=power_transformer_data["event"],
        entry=power_transformer_data["entry"],
    )
    assert_allclose(distribution.get_params(), expected_params, rtol=1e-3)


def test_negative_log(
    distribution_likelihood: LifetimeLikelihood[LifetimeDistribution],
):
    params = distribution_likelihood.model.get_params().copy()
    assert isinstance(distribution_likelihood.negative_log(params), float)


def test_jac_negative_log(
    distribution_likelihood: LifetimeLikelihood[LifetimeDistribution],
):
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
    def test_moment(self, equilibrium_distribution: LifetimeDistribution):
        assert equilibrium_distribution.moment(1).shape == ()
        assert equilibrium_distribution.moment(2).shape == ()

    @pytest.mark.xfail
    def test_mean(self, equilibrium_distribution: LifetimeDistribution):
        assert equilibrium_distribution.mean().shape == ()

    @pytest.mark.xfail
    def test_var(self, equilibrium_distribution: LifetimeDistribution):
        assert equilibrium_distribution.var().shape == ()

    @pytest.mark.xfail
    def test_median(self, equilibrium_distribution: LifetimeDistribution):
        assert equilibrium_distribution.median().shape == ()
