from typing import Literal, TypeAlias

import numpy as np
import pytest
from numpy.testing import assert_allclose
from optype.numpy import Array1D, ArrayND

from relife.lifetime_models._distributions import LifetimeDistribution
from relife.utils import to_numpy_float64

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


class TestBroadcasting:
    @pytest.mark.parametrize(
        "time_or_probability",
        [np.ones(()) * 0.5, np.ones((1, 2)) * 0.5, np.ones((3, 5)) * 0.5],
        ids=lambda x: f"{x.shape}",
    )
    @pytest.mark.parametrize(
        "method",
        [
            "sf",
            "hf",
            "chf",
            "cdf",
            "ppf",
            "pdf",
            "dhf",
            "ppf",
            "ichf",
            "isf",
        ],
    )
    def test_probability_functions(
        self,
        distribution: LifetimeDistribution,
        method: Literal[
            "sf",
            "hf",
            "chf",
            "cdf",
            "ppf",
            "pdf",
            "dhf",
            "ppf",
            "ichf",
            "isf",
        ],
        time_or_probability: ArrayND[np.float64],
    ):
        assert (
            getattr(distribution, method)(time_or_probability).shape
            == time_or_probability.shape
        )

    @pytest.mark.parametrize(
        "time",
        [np.ones(()), np.ones((1, 2)), np.ones((3, 5))],
        ids=lambda x: f"{x.shape}",
    )
    @pytest.mark.parametrize(
        "method",
        [
            "jac_sf",
            "jac_chf",
            "jac_cdf",
            "jac_pdf",
        ],
    )
    def test_jac_functions(
        self,
        distribution: LifetimeDistribution,
        method: Literal[
            "jac_sf",
            "jac_chf",
            "jac_cdf",
            "jac_pdf",
        ],
        time: ArrayND[np.float64],
    ):
        assert (
            getattr(distribution, method)(time).shape
            == (distribution.get_params().size,) + time.shape
        )

    @pytest.mark.parametrize(
        "size",
        [(), 3, (1, 2), (3, 4, 5)],
        ids=lambda x: f"{x}",
    )
    def test_rvs(
        self,
        distribution: LifetimeDistribution,
        size: int | tuple[int, ...],
    ):
        expected_shape = (size,) if isinstance(size, int) else size
        assert distribution.rvs(size).shape == expected_shape

    def test_moment(self, distribution: LifetimeDistribution):
        assert distribution.moment(1).shape == ()
        assert distribution.moment(2).shape == ()

    def test_mean(self, distribution: LifetimeDistribution):
        assert distribution.mean().shape == ()

    def test_var(self, distribution: LifetimeDistribution):
        assert distribution.var().shape == ()

    def test_median(self, distribution: LifetimeDistribution):
        assert distribution.median().shape == ()

    @pytest.mark.parametrize(
        "a",
        [
            np.ones(()) * 2,
            np.ones((1, 2)) * 2,
            np.ones((3, 2)) * 2,
            np.ones((3, 5)) * 2,
        ],
        ids=lambda x: f"{x.shape}",
    )
    @pytest.mark.parametrize(
        "b",
        [
            np.ones(()) * 8,
            np.ones((1, 2)) * 8,
            np.ones((3, 2)) * 8,
            np.ones((3, 5)) * 8,
        ],
        ids=lambda x: f"{x.shape}",
    )
    def test_ls_integrate(
        self,
        distribution: LifetimeDistribution,
        a: ArrayND[np.float64],
        b: ArrayND[np.float64],
    ):
        try:
            expected_shape = np.broadcast_shapes(a.shape, b.shape)
        except ValueError:
            with pytest.raises(
                ValueError, match=r"(shape mismatch*|operands could not be broadcast*)"
            ):
                _ = distribution.ls_integrate(np.ones_like, a, b)
        else:
            integration = distribution.ls_integrate(np.ones_like, a, b)
            assert integration.shape == expected_shape

    def test_apply_condition(
        self,
        distribution: LifetimeDistribution,
        method,
        ar: ArrayND[np.float64] | None,
        a0: ArrayND[np.float64] | None,
        time: ArrayND[np.float64],
    ):
        try:
            ar_shape = ar.shape if ar else ()
            a0_shape = a0.shape if a0 else ()
            expected_shape = np.broadcast_shapes(ar_shape, a0_shape, time.shape)
        except ValueError:
            with pytest.raises(
                ValueError, match=r"(shape mismatch*|operands could not be broadcast*)"
            ):
                _ = getattr(distribution.apply_condition(ar=ar, a0=a0), method)(time)
        else:
            assert (
                getattr(distribution.apply_condition(ar=ar, a0=a0), method)(time).shape
                == expected_shape
            )


def test_sf_values(distribution: LifetimeDistribution):
    assert_allclose(
        distribution.sf(np.full((3, 5), distribution.median())),
        np.full((3, 5), 0.5),
        rtol=1e-3,
    )


def test_isf_values(distribution: LifetimeDistribution):
    assert_allclose(
        distribution.isf(np.full((3, 5), 0.5)),
        np.full((3, 5), distribution.median()),
    )


def test_ls_integrate_values(
    distribution: LifetimeDistribution,
):
    a = np.ones((3, 5)) * 2
    b = np.ones((3, 5)) * 8
    integration = distribution.ls_integrate(np.ones_like, a, b, deg=100)
    assert_allclose(
        integration,
        distribution.cdf(b) - distribution.cdf(a),
    )

    def func(x: ST | NumpyST | ArrayND[NumpyST]) -> np.float64 | ArrayND[np.float64]:
        return to_numpy_float64(x)

    integration = distribution.ls_integrate(
        func,
        np.zeros_like(a),
        np.full_like(b, np.inf),
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
    distribution: LifetimeDistribution, power_transformer_data: Array1D[np.void]
):
    likelihood = distribution.init_likelihood(
        power_transformer_data["time"],
        event=power_transformer_data["event"],
        entry=power_transformer_data["entry"],
    )
    assert isinstance(likelihood.negative_log(distribution.get_params()), float)


def test_jac_negative_log(
    distribution: LifetimeDistribution, power_transformer_data: Array1D[np.void]
):
    likelihood = distribution.init_likelihood(
        power_transformer_data["time"],
        event=power_transformer_data["event"],
        entry=power_transformer_data["entry"],
    )
    params = distribution.get_params()
    assert likelihood.jac_negative_log(params).shape == (params.size,)


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
