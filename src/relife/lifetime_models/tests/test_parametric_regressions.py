import numpy as np
from optype.numpy import ArrayND
from pytest import approx
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
from relife.lifetime_models.tests.test_conditional_models import expected_shape


def test_covar_effect():
    covar_effect = LinearCovarEffect(coefficients=(2.4, 5.5))
    z1 = np.array([1, 2, 3])
    z2 = np.array([0.8, 0.7, 0.5])
    assert covar_effect.g(z1, z2) == approx(np.exp(2.4 * z1 + 5.5 * z2))
    assert covar_effect.jac_g(z1, z2)[0] == approx(z1 * np.exp(2.4 * z1 + 5.5 * z2))
    assert covar_effect.jac_g(z1, z2)[1] == approx(z2 * np.exp(2.4 * z1 + 5.5 * z2))


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
        time, *covar = time_covar
        assert (
            regression.sf(time, *covar).shape
            == np.broadcast_arrays(time, *covar)[0].shape
        )

    def test_hf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        time, *covar = time_covar
        assert (
            regression.hf(time, *covar).shape
            == np.broadcast_arrays(time, *covar)[0].shape
        )

    def test_chf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        time, *covar = time_covar
        assert (
            regression.chf(time, *covar).shape
            == np.broadcast_arrays(time, *covar)[0].shape
        )

    def test_cdf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        time, *covar = time_covar
        assert (
            regression.cdf(time, *covar).shape
            == np.broadcast_arrays(time, *covar)[0].shape
        )

    def test_pdf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        time, *covar = time_covar
        assert (
            regression.pdf(time, *covar).shape
            == np.broadcast_arrays(time, *covar)[0].shape
        )

    def test_ppf(
        self,
        regression: ParametricLifetimeRegression,
        probability_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        probability, *covar = probability_covar
        assert (
            regression.ppf(probability, *covar).shape
            == np.broadcast_arrays(probability, *covar)[0].shape
        )

    def test_ichf(
        self,
        regression: ParametricLifetimeRegression,
        probability_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        probability, *covar = probability_covar
        assert (
            regression.ichf(probability, *covar).shape
            == np.broadcast_arrays(probability, *covar)[0].shape
        )

    def test_isf(
        self,
        regression: ParametricLifetimeRegression,
        probability_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        probability, *covar = probability_covar
        assert (
            regression.isf(probability, *covar).shape
            == np.broadcast_arrays(probability, *covar)[0].shape
        )

    def test_dhf(
        self,
        regression: ParametricLifetimeRegression,
        time_covar: tuple[
            ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
        ],
    ):
        time, *covar = time_covar
        assert (
            regression.dhf(time, *covar).shape
            == np.broadcast_arrays(time, *covar)[0].shape
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
        ls_integrate = regression.ls_integrate(
            np.ones_like,
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
    _, *covar = time_covar
    median = regression.median(*covar)
    assert regression.sf(median) == approx(np.full_like(median, 0.5), rel=1e-3)


def test_isf_values(
    regression: ParametricLifetimeRegression,
    time_probability: tuple[
        ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
    ],
):
    _, *covar = time_probability
    median = regression.median(*covar)
    assert regression.isf(np.full_like(median, 0.5), *covar) == approx(median, rel=1e-3)


def test_ls_integrate(regression, integration_bound_a, integration_bound_b, covar):
    # integral_a^b dF(x)
    integration = regression.ls_integrate(
        np.ones_like, integration_bound_a, integration_bound_b, covar, deg=100
    )
    assert integration.shape == expected_shape(
        a=integration_bound_a, b=integration_bound_b, covar=covar
    )
    assert integration == approx(
        regression.cdf(integration_bound_b, covar)
        - regression.cdf(integration_bound_a, covar)
    )
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate(
        lambda x: x,
        np.zeros_like(integration_bound_a),
        np.full_like(integration_bound_b, np.inf),
        covar,
        deg=100,
    )
    assert integration == approx(
        np.broadcast_to(
            regression.mean(covar),
            expected_shape(a=integration_bound_a, b=integration_bound_b, covar=covar),
        ),
        rel=1e-3,
    )


def test_aft_pph_weibull_eq(insulator_string_data):
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
        covar_data,
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
    )
    weibull_pph = ParametricProportionalHazard(Weibull()).fit(
        insulator_string_data["time"],
        covar_data,
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
    )

    assert weibull_pph.baseline.get_params() == approx(
        weibull_aft.baseline.get_params(), rel=1e-3
    )
    assert weibull_pph.covar_effect.get_params() == approx(
        -weibull_aft.baseline.get_params()[0] * weibull_aft.covar_effect.get_params(),
        rel=1e-3,
    )


def test_negative_log(regression_likelihood):
    params = regression_likelihood.model.get_params().copy()
    assert isinstance(regression_likelihood.negative_log(params), float)


def test_jac_negative_log(regression_likelihood):
    params = regression_likelihood.model.get_params().copy()
    assert regression_likelihood.jac_negative_log(params).shape == (params.size,)


class TestFrozenRegression:
    def test_rvs(self, regression, covar, rvs_size):
        frozen_model = regression.freeze(covar)
        assert frozen_model.rvs(rvs_size).shape == rvs_expected_shape(
            rvs_size, covar=covar
        )

    def test_sf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.sf(time).shape == expected_shape(time=time, covar=covar)

    def test_hf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.hf(time).shape == expected_shape(time=time, covar=covar)

    def test_chf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.chf(time).shape == expected_shape(time=time, covar=covar)

    def test_cdf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.cdf(time).shape == expected_shape(time=time, covar=covar)

    def test_pdf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.pdf(time).shape == expected_shape(time=time, covar=covar)

    def test_ppf(self, regression, probability, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.ppf(probability).shape == expected_shape(
            probability=probability, covar=covar
        )

    def test_ichf(self, regression, probability, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.ichf(probability).shape == expected_shape(
            probability=probability, covar=covar
        )

    def test_isf(self, regression, probability, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.isf(probability).shape == expected_shape(
            probability=probability, covar=covar
        )
        assert frozen_model.isf(
            np.full(expected_shape(probability=probability, covar=covar), 0.5)
        ) == approx(
            np.broadcast_to(
                frozen_model.median(),
                expected_shape(probability=probability, covar=covar),
            )
        )

    def test_dhf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.dhf(time).shape == expected_shape(time=time, covar=covar)

    def test_jac_sf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        nb_params = frozen_model.get_params().size
        assert frozen_model.jac_sf(time).shape == (nb_params,) + expected_shape(
            time=time, covar=covar
        )

    def test_jac_hf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        nb_params = frozen_model.get_params().size
        assert frozen_model.jac_hf(time).shape == (nb_params,) + expected_shape(
            time=time, covar=covar
        )

    def test_jac_chf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        nb_params = frozen_model.get_params().size
        assert frozen_model.jac_chf(time).shape == (nb_params,) + expected_shape(
            time=time, covar=covar
        )

    def test_jac_cdf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        nb_params = frozen_model.get_params().size
        assert frozen_model.jac_cdf(time).shape == (nb_params,) + expected_shape(
            time=time, covar=covar
        )

    def test_jac_pdf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        nb_params = frozen_model.get_params().size
        assert frozen_model.jac_pdf(time).shape == (nb_params,) + expected_shape(
            time=time, covar=covar
        )
