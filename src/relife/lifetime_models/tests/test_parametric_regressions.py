from typing import TypeAlias

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from optype.numpy import Array1D, ArrayND
from scipy.stats import boxcox, zscore

from relife.lifetime_models import (
    ParametricAcceleratedFailureTime,
    ParametricProportionalHazard,
    Weibull,
)
from relife.lifetime_models._parametric_regressions import (
    LinearCovarEffect,
    ParametricLifetimeRegression,
)

from .utils import generate_shapes

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


def test_covar_effect():
    covar_effect = LinearCovarEffect(coefficients=(2.4, 5.5))
    z1 = np.array([1, 2, 3])
    z2 = np.array([0.8, 0.7, 0.5])
    assert_equal(covar_effect.g(z1, z2), np.exp(2.4 * z1 + 5.5 * z2))
    assert_equal(covar_effect.jac_g(z1, z2)[0], z1 * np.exp(2.4 * z1 + 5.5 * z2))
    assert_equal(covar_effect.jac_g(z1, z2)[1], z2 * np.exp(2.4 * z1 + 5.5 * z2))


class TestBroadcasting:
    @pytest.mark.parametrize(
        "method",
        ["sf", "hf", "chf", "cdf", "pdf", "dhf", "isf", "ichf", "ppf"],
    )
    @pytest.mark.parametrize(
        "shape, z1_shape, z2_shape",
        generate_shapes(3, 2),
    )
    def test_prob_func(
        self,
        regression: ParametricLifetimeRegression,
        method: str,
        shape: tuple[int] | tuple[int, int],
        z1_shape: tuple[int] | tuple[int, int],
        z2_shape: tuple[int] | tuple[int, int],
    ):
        assert getattr(regression, method)(
            np.ones(shape) * 0.5, np.ones(z1_shape), np.ones(z2_shape)
        ).shape == np.broadcast_shapes(shape, z1_shape, z2_shape)

    @pytest.mark.parametrize(
        "method",
        ["sf", "hf", "chf", "cdf", "pdf", "isf", "ichf", "ppf"],
    )
    @pytest.mark.parametrize(
        "a0_shape, shape, z1_shape, z2_shape",
        generate_shapes(4, 2),
    )
    def test_a0_prob_func(
        self,
        regression: ParametricLifetimeRegression,
        method: str,
        a0_shape: tuple[int] | tuple[int, int],
        shape: tuple[int] | tuple[int, int],
        z1_shape: tuple[int] | tuple[int, int],
        z2_shape: tuple[int] | tuple[int, int],
    ):
        assert getattr(regression.apply_condition(a0=np.ones(a0_shape) * 0.3), method)(
            np.ones(shape) * 0.5,
            np.ones(z1_shape),
            np.ones(z2_shape),
        ).shape == np.broadcast_shapes(a0_shape, shape, z1_shape, z2_shape)

    @pytest.mark.parametrize(
        "method",
        ["sf", "hf", "chf", "cdf", "pdf", "isf", "ichf", "ppf"],
    )
    @pytest.mark.parametrize(
        "ar_shape, shape, z1_shape, z2_shape",
        generate_shapes(4, 2),
    )
    def test_ar_prob_func(
        self,
        regression: ParametricLifetimeRegression,
        method: str,
        ar_shape: tuple[int] | tuple[int, int],
        shape: tuple[int] | tuple[int, int],
        z1_shape: tuple[int] | tuple[int, int],
        z2_shape: tuple[int] | tuple[int, int],
    ):
        assert getattr(regression.apply_condition(ar=np.ones(ar_shape) * 0.3), method)(
            np.ones(shape) * 0.5,
            np.ones(z1_shape),
            np.ones(z2_shape),
        ).shape == np.broadcast_shapes(ar_shape, shape, z1_shape, z2_shape)

    @pytest.mark.parametrize(
        "method",
        ["jac_sf", "jac_chf", "jac_cdf", "jac_pdf"],
    )
    @pytest.mark.parametrize(
        "time_shape, z1_shape, z2_shape",
        generate_shapes(3, 2),
    )
    def test_jac_functions(
        self,
        regression: ParametricLifetimeRegression,
        method: str,
        time_shape: tuple[int] | tuple[int, int],
        z1_shape: tuple[int] | tuple[int, int],
        z2_shape: tuple[int] | tuple[int, int],
    ):
        assert getattr(regression, method)(
            np.ones(time_shape), np.ones(z1_shape), np.ones(z2_shape)
        ).shape == (regression.get_params().size,) + np.broadcast_shapes(
            time_shape, z1_shape, z2_shape
        )

    @pytest.mark.parametrize(
        "size, z1_shape, z2_shape",
        generate_shapes(3, 2),
    )
    def test_rvs(
        self,
        regression: ParametricLifetimeRegression,
        size: tuple[int] | tuple[int, int],
        z1_shape: tuple[int] | tuple[int, int],
        z2_shape: tuple[int] | tuple[int, int],
    ):
        assert regression.rvs(
            size, np.ones(z1_shape), np.ones(z2_shape), seed=1
        ).shape == np.broadcast_shapes(size, z1_shape, z2_shape)

    @pytest.mark.parametrize(
        "method",
        ["mean", "var", "median"],
    )
    @pytest.mark.parametrize(
        "z1_shape, z2_shape",
        generate_shapes(2, 2),
    )
    def test_moment_func(
        self,
        regression: ParametricLifetimeRegression,
        method: str,
        z1_shape: tuple[int] | tuple[int, int],
        z2_shape: tuple[int] | tuple[int, int],
    ):
        assert getattr(regression, method)(
            np.ones(z1_shape), np.ones(z2_shape)
        ).shape == np.broadcast_shapes(z1_shape, z2_shape)

    @pytest.mark.parametrize(
        "a_shape, b_shape, z1_shape, z2_shape",
        generate_shapes(4, 2),
    )
    def test_ls_integrate(
        self,
        regression: ParametricLifetimeRegression,
        a_shape: tuple[int] | tuple[int, int],
        b_shape: tuple[int] | tuple[int, int],
        z1_shape: tuple[int] | tuple[int, int],
        z2_shape: tuple[int] | tuple[int, int],
    ):
        integration = regression.ls_integrate(
            np.ones_like,
            np.ones(a_shape) * 2.0,
            np.ones(b_shape) * 8.0,
            np.ones(z1_shape),
            np.ones(z2_shape),
        )
        assert integration.shape == np.broadcast_shapes(
            a_shape, b_shape, z1_shape, z2_shape
        )

    @pytest.mark.parametrize(
        "a0_shape, a_shape, b_shape, z1_shape, z2_shape",
        generate_shapes(5, 2),
    )
    def test_a0_ls_integrate(
        self,
        regression: ParametricLifetimeRegression,
        a0_shape: tuple[int] | tuple[int, int],
        a_shape: tuple[int] | tuple[int, int],
        b_shape: tuple[int] | tuple[int, int],
        z1_shape: tuple[int] | tuple[int, int],
        z2_shape: tuple[int] | tuple[int, int],
    ):
        integration = regression.apply_condition(
            a0=np.ones(a0_shape) * 0.3
        ).ls_integrate(
            np.ones_like,
            np.ones(a_shape) * 2.0,
            np.ones(b_shape) * 8.0,
            np.ones(z1_shape),
            np.ones(z2_shape),
        )
        assert integration.shape == np.broadcast_shapes(
            a0_shape, a_shape, b_shape, z1_shape, z2_shape
        )

    @pytest.mark.parametrize(
        "ar_shape, a_shape, b_shape, z1_shape, z2_shape",
        generate_shapes(5, 2),
    )
    def test_ar_ls_integrate(
        self,
        regression: ParametricLifetimeRegression,
        ar_shape: tuple[int] | tuple[int, int],
        a_shape: tuple[int] | tuple[int, int],
        b_shape: tuple[int] | tuple[int, int],
        z1_shape: tuple[int] | tuple[int, int],
        z2_shape: tuple[int] | tuple[int, int],
    ):
        integration = regression.apply_condition(
            ar=np.ones(ar_shape) * 3.0
        ).ls_integrate(
            np.ones_like,
            np.ones(a_shape) * 2.0,
            np.ones(b_shape) * 8.0,
            np.ones(z1_shape),
            np.ones(z2_shape),
        )
        assert integration.shape == np.broadcast_shapes(
            ar_shape, a_shape, b_shape, z1_shape, z2_shape
        )


def test_sf_values(
    regression: ParametricLifetimeRegression,
):
    median = regression.median(np.ones((3, 5)), np.ones((3, 5)))
    assert_allclose(
        regression.sf(median, np.ones((3, 5)), np.ones((3, 5))),
        np.full_like(median, 0.5),
        rtol=1e-3,
    )


def test_isf_values(
    regression: ParametricLifetimeRegression,
):
    median = regression.median(np.ones((3, 5)), np.ones((3, 5)))
    assert_allclose(
        regression.isf(np.full_like(median, 0.5), np.ones((3, 5)), np.ones((3, 5))),
        median,
        rtol=1e-3,
    )


def test_ls_integrate_values(
    regression: ParametricLifetimeRegression,
):
    a = np.ones((3, 5)) * 2
    b = np.ones((3, 5)) * 8
    covar_1 = np.ones((3, 5))
    covar_2 = np.ones((3, 5))
    integration = regression.ls_integrate(np.ones_like, a, b, covar_1, covar_2, deg=100)
    assert_allclose(
        integration,
        regression.cdf(b, covar_1, covar_2) - regression.cdf(a, covar_1, covar_2),
    )

    def func(
        x: ST | NumpyST | ArrayND[NumpyST], *args: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return np.broadcast_arrays(x, *args)[0]

    integration = regression.ls_integrate(
        func,
        0.0,
        np.inf,
        covar_1,
        covar_2,
        deg=100,
    )
    assert_allclose(integration, regression.mean(covar_1, covar_2))


def test_aft_pph_weibull_eq(insulator_string_data: Array1D[np.void]):
    covar_1 = zscore(boxcox(insulator_string_data["pHCl"])[0])
    covar_2 = zscore(boxcox(insulator_string_data["pH2SO4"])[0])
    covar_3 = zscore(boxcox(insulator_string_data["HNO3"])[0])
    weibull_aft = ParametricAcceleratedFailureTime(Weibull()).fit(
        insulator_string_data["time"],
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
        covar=(covar_1, covar_2, covar_3),
    )
    weibull_pph = ParametricProportionalHazard(Weibull()).fit(
        insulator_string_data["time"],
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
        covar=(covar_1, covar_2, covar_3),
    )

    assert_allclose(
        weibull_pph.baseline.get_params(), weibull_aft.baseline.get_params(), rtol=1e-3
    )
    assert_allclose(
        weibull_pph.covar_effect.get_params(),
        -weibull_aft.baseline.get_params()[0] * weibull_aft.covar_effect.get_params(),
        rtol=1e-3,
    )


def test_negative_log(
    regression: ParametricLifetimeRegression, insulator_string_data: Array1D[np.void]
):
    covar_1 = zscore(boxcox(insulator_string_data["pHCl"])[0])
    covar_2 = zscore(boxcox(insulator_string_data["pH2SO4"])[0])
    covar_3 = zscore(boxcox(insulator_string_data["HNO3"])[0])
    likelihood = regression.init_likelihood(
        insulator_string_data["time"],
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
        covar=(covar_1, covar_2, covar_3),
    )
    params = likelihood.model.get_params() * 10.0
    print(params)
    assert isinstance(likelihood.negative_log(params), float)


def test_jac_negative_log(
    regression: ParametricLifetimeRegression, insulator_string_data: Array1D[np.void]
):
    covar_1 = zscore(boxcox(insulator_string_data["pHCl"])[0])
    covar_2 = zscore(boxcox(insulator_string_data["pH2SO4"])[0])
    covar_3 = zscore(boxcox(insulator_string_data["HNO3"])[0])
    likelihood = regression.init_likelihood(
        insulator_string_data["time"],
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
        covar=(covar_1, covar_2, covar_3),
    )
    params = likelihood.model.get_params() * 10.0
    assert likelihood.jac_negative_log(params).shape == (params.size,)
