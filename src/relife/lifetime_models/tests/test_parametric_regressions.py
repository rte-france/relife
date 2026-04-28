from typing import TypeAlias

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from optype.numpy import Array1D, ArrayND
from scipy.stats import boxcox, zscore

from relife.lifetime_models import (
    LifetimeLikelihood,
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
    def test_rvs(
        self,
        regression: ParametricLifetimeRegression,
        rvs_size_covar: tuple[
            tuple[int, ...], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        rvs_size, covar_1, covar_2 = rvs_size_covar
        assert regression.rvs(rvs_size, covar_1, covar_2).shape == np.broadcast_shapes(
            rvs_size, covar_1.shape, covar_2.shape
        )

    def test_sf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        time, covar_1, covar_2 = time_covar
        assert (
            regression.sf(time, covar_1, covar_2).shape
            == np.broadcast_arrays(time, covar_1, covar_2)[0].shape
        )

    def test_hf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        time, covar_1, covar_2 = time_covar
        assert (
            regression.hf(time, covar_1, covar_2).shape
            == np.broadcast_arrays(time, covar_1, covar_2)[0].shape
        )

    def test_chf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        time, covar_1, covar_2 = time_covar
        assert (
            regression.chf(time, covar_1, covar_2).shape
            == np.broadcast_arrays(time, covar_1, covar_2)[0].shape
        )

    def test_cdf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        time, covar_1, covar_2 = time_covar
        assert (
            regression.cdf(time, covar_1, covar_2).shape
            == np.broadcast_arrays(time, covar_1, covar_2)[0].shape
        )

    def test_pdf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        time, covar_1, covar_2 = time_covar
        assert (
            regression.pdf(time, covar_1, covar_2).shape
            == np.broadcast_arrays(time, covar_1, covar_2)[0].shape
        )

    def test_ppf(
        self,
        regression: ParametricLifetimeRegression,
        probability_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        probability, covar_1, covar_2 = probability_covar
        assert (
            regression.ppf(probability, covar_1, covar_2).shape
            == np.broadcast_arrays(probability, covar_1, covar_2)[0].shape
        )

    def test_ichf(
        self,
        regression: ParametricLifetimeRegression,
        probability_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        probability, covar_1, covar_2 = probability_covar
        assert (
            regression.ichf(probability, covar_1, covar_2).shape
            == np.broadcast_arrays(probability, covar_1, covar_2)[0].shape
        )

    def test_isf(
        self,
        regression: ParametricLifetimeRegression,
        probability_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        probability, covar_1, covar_2 = probability_covar
        assert (
            regression.isf(probability, covar_1, covar_2).shape
            == np.broadcast_arrays(probability, covar_1, covar_2)[0].shape
        )

    def test_dhf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        time, covar_1, covar_2 = time_covar
        assert (
            regression.dhf(time, covar_1, covar_2).shape
            == np.broadcast_arrays(time, covar_1, covar_2)[0].shape
        )

    def test_jac_sf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        nb_params = regression.get_params().size
        time, covar_1, covar_2 = time_covar
        assert regression.jac_sf(time, covar_1, covar_2).shape == (
            nb_params,
        ) + np.broadcast_shapes(time.shape, covar_1.shape, covar_2.shape)

    def test_jac_hf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        nb_params = regression.get_params().size
        time, covar_1, covar_2 = time_covar
        assert regression.jac_hf(time, covar_1, covar_2).shape == (
            nb_params,
        ) + np.broadcast_shapes(time.shape, covar_1.shape, covar_2.shape)

    def test_jac_chf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        nb_params = regression.get_params().size
        time, covar_1, covar_2 = time_covar
        assert regression.jac_chf(time, covar_1, covar_2).shape == (
            nb_params,
        ) + np.broadcast_shapes(time.shape, covar_1.shape, covar_2.shape)

    def test_jac_cdf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        nb_params = regression.get_params().size
        time, covar_1, covar_2 = time_covar
        assert regression.jac_cdf(time, covar_1, covar_2).shape == (
            nb_params,
        ) + np.broadcast_shapes(time.shape, covar_1.shape, covar_2.shape)

    def test_jac_pdf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        nb_params = regression.get_params().size
        time, covar_1, covar_2 = time_covar
        assert regression.jac_pdf(time, covar_1, covar_2).shape == (
            nb_params,
        ) + np.broadcast_shapes(time.shape, covar_1.shape, covar_2.shape)

    def test_ls_integrate(
        self,
        regression: ParametricLifetimeRegression,
        a_b_covar: tuple[
            ArrayND[np.float64],
            ArrayND[np.float64],
            ArrayND[np.float64],
            ArrayND[np.float64],
        ],
    ):
        a, b, covar_1, covar_2 = a_b_covar
        expected_shape = np.broadcast_shapes(
            a.shape, b.shape, covar_1.shape, covar_2.shape
        )

        def func(
            x: ST | NumpyST | ArrayND[NumpyST], *args: ST | NumpyST | ArrayND[NumpyST]
        ) -> np.float64 | ArrayND[np.float64]:
            return np.ones_like(np.broadcast_arrays(x, *args)[0], dtype=np.float64)

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
    time_covar: tuple[ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]],
):
    _, covar_1, covar_2 = time_covar
    median = regression.median(covar_1, covar_2)
    assert_allclose(regression.sf(median), np.full_like(median, 0.5), rtol=1e-3)


def test_isf_values(
    regression: ParametricLifetimeRegression,
    time_probability: tuple[
        ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
    ],
):
    _, covar_1, covar_2 = time_probability
    median = regression.median(covar_1, covar_2)
    assert_allclose(
        regression.isf(np.full_like(median, 0.5), covar_1, covar_2), median, rtol=1e-3
    )


def test_ls_integrate_values(
    regression: ParametricLifetimeRegression,
    a_b_covar: tuple[
        ArrayND[np.float64],
        ArrayND[np.float64],
        ArrayND[np.float64],
        ArrayND[np.float64],
    ],
):
    a, b, covar_1, covar_2 = a_b_covar
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
    covar_data = zscore(
        np.column_stack(
            (
                boxcox(insulator_string_data["pHCl"])[0],
                boxcox(insulator_string_data["pH2SO4"])[0],
                boxcox(insulator_string_data["HNO3"])[0],
            )
        )
    )
    weibull_aft = ParametricAcceleratedFailureTime(Weibull()).fit(
        insulator_string_data["time"],
        covar=covar_data,
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
    )
    weibull_pph = ParametricProportionalHazard(Weibull()).fit(
        insulator_string_data["time"],
        covar=covar_data,
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
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
    regression_likelihood: LifetimeLikelihood[ParametricLifetimeRegression],
):
    params = regression_likelihood.model.get_params().copy()
    assert isinstance(regression_likelihood.negative_log(params), float)


def test_jac_negative_log(
    regression_likelihood: LifetimeLikelihood[ParametricLifetimeRegression],
):
    params = regression_likelihood.model.get_params().copy()
    assert regression_likelihood.jac_negative_log(params).shape == (params.size,)
