from typing import Literal, TypeAlias

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
        "time_or_probability",
        [np.ones(()) * 0.5, np.ones((1, 2)) * 0.5, np.ones((3, 5)) * 0.5],
    )
    @pytest.mark.parametrize(
        "covar_1",
        [np.ones(()), np.ones((1, 2)), np.ones((3, 5))],
    )
    @pytest.mark.parametrize(
        "covar_2",
        [np.ones(()), np.ones((1, 2)), np.ones((3, 5))],
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
        regression: ParametricLifetimeRegression,
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
        covar_1: ArrayND[np.float64],
        covar_2: ArrayND[np.float64],
    ):
        try:
            expected_shape = np.broadcast_shapes(
                time_or_probability.shape, covar_1.shape, covar_2.shape
            )
        except ValueError:
            with pytest.raises(
                ValueError, match=r"(shape mismatch*|operands could not be broadcast*)"
            ):
                _ = getattr(regression, method)(time_or_probability, covar_1, covar_2)
        else:
            assert (
                getattr(regression, method)(time_or_probability, covar_1, covar_2).shape
                == expected_shape
            )

    @pytest.mark.parametrize(
        "time",
        [np.ones(()), np.ones((1, 2)), np.ones((3, 5))],
    )
    @pytest.mark.parametrize(
        "covar_1",
        [np.ones(()), np.ones((1, 2)), np.ones((3, 5))],
    )
    @pytest.mark.parametrize(
        "covar_2",
        [np.ones(()), np.ones((1, 2)), np.ones((3, 5))],
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
        regression: ParametricLifetimeRegression,
        method: Literal[
            "jac_sf",
            "jac_chf",
            "jac_cdf",
            "jac_pdf",
        ],
        time: ArrayND[np.float64],
        covar_1: ArrayND[np.float64],
        covar_2: ArrayND[np.float64],
    ):
        try:
            expected_shape = np.broadcast_shapes(
                time.shape, covar_1.shape, covar_2.shape
            )
        except ValueError:
            with pytest.raises(
                ValueError, match=r"(shape mismatch*|operands could not be broadcast*)"
            ):
                _ = getattr(regression, method)(time, covar_1, covar_2)
        else:
            assert (
                getattr(regression, method)(time, covar_1, covar_2).shape
                == (regression.get_params().size,) + expected_shape
            )

    @pytest.mark.parametrize(
        "size",
        [(), 3, (1, 2), (3, 4, 5)],
        ids=lambda x: f"{x}",
    )
    @pytest.mark.parametrize(
        "covar_1",
        [np.ones(()), np.ones((1, 2)), np.ones((3, 2)), np.ones((3, 5))],
    )
    @pytest.mark.parametrize(
        "covar_2",
        [np.ones(()), np.ones((1, 2)), np.ones((3, 2)), np.ones((3, 5))],
    )
    def test_rvs(
        self,
        regression: ParametricLifetimeRegression,
        size: int | tuple[int, ...],
        covar_1: ArrayND[np.float64],
        covar_2: ArrayND[np.float64],
    ):
        try:
            expected_shape = np.broadcast_shapes(size, covar_1.shape, covar_2.shape)
        except ValueError:
            with pytest.raises(
                ValueError, match=r"(shape mismatch*|operands could not be broadcast*)"
            ):
                _ = regression.rvs(size, covar_1, covar_2, seed=1)
        else:
            assert (
                regression.rvs(size, covar_1, covar_2, seed=1).shape == expected_shape
            )

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
    @pytest.mark.parametrize(
        "covar_1",
        [np.ones(()), np.ones((1, 2)), np.ones((3, 2)), np.ones((3, 5))],
    )
    @pytest.mark.parametrize(
        "covar_2",
        [np.ones(()), np.ones((1, 2)), np.ones((3, 2)), np.ones((3, 5))],
    )
    def test_ls_integrate(
        self,
        regression: ParametricLifetimeRegression,
        a: ArrayND[np.float64],
        b: ArrayND[np.float64],
        covar_1: ArrayND[np.float64],
        covar_2: ArrayND[np.float64],
    ):
        def func(
            x: ST | NumpyST | ArrayND[NumpyST], *args: ST | NumpyST | ArrayND[NumpyST]
        ) -> np.float64 | ArrayND[np.float64]:
            return np.ones_like(np.broadcast_arrays(x, *args)[0], dtype=np.float64)

        try:
            expected_shape = np.broadcast_shapes(
                a.shape, b.shape, covar_1.shape, covar_2.shape
            )
        except ValueError:
            with pytest.raises(
                ValueError, match=r"(shape mismatch*|operands could not be broadcast*)"
            ):
                _ = regression.ls_integrate(
                    func,
                    a,
                    b,
                    covar_1,
                    covar_2,
                )
        else:
            ls_integrate = regression.ls_integrate(
                func,
                a,
                b,
                covar_1,
                covar_2,
            )
            assert ls_integrate.shape == expected_shape


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
    params = regression.get_params()
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
    params = regression.get_params()
    assert likelihood.jac_negative_log(params).shape == (params.size,)
