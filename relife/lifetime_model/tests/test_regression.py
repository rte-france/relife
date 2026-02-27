# pyright: basic

import numpy as np
from pytest import approx
from scipy.stats import boxcox, zscore

from relife.lifetime_model import (
    ParametricAcceleratedFailureTime,
    ParametricProportionalHazard,
    Weibull,
)
from relife.lifetime_model._regression import LinearCovarEffect
from relife.lifetime_model._semi_parametric import SemiParametricProportionalHazard


def expected_shape(**kwargs):
    def shape_contrib(**_kwargs):
        yield ()  # yield at least (), in case kwargs is empty
        for k, v in _kwargs.items():
            match k:
                case "covar" if v.ndim == 2:
                    yield v.shape[0], 1
                case "covar" if v.ndim < 2:
                    yield ()
                case _:
                    yield v.shape

    return np.broadcast_shapes(*tuple(shape_contrib(**kwargs)))


def rvs_expected_shape(size, **kwargs):
    out_shape = expected_shape(**kwargs)
    if size != 1:
        return np.broadcast_shapes(out_shape, (size,))
    return out_shape


def test_covar_effect():
    """
    covar : () or (nb_coef,)
    => g : ()
    => jac_g : (nb_coef,)

    covar : (m, nb_coef)
    => g : (m, 1)
    => jac_g : (nb_coef, m, 1)
    """

    covar_effect = LinearCovarEffect(coefficients=(2.4, 5.5))
    z1 = np.array([1, 2, 3])
    z2 = np.array([0.8, 0.7, 0.5])
    assert covar_effect.g(np.column_stack((z1, z2))) == approx(
        np.exp(2.4 * z1 + 5.5 * z2).reshape(-1, 1)
    )
    assert covar_effect.jac_g(np.column_stack((z1, z2)))[0] == approx(
        (z1 * np.exp(2.4 * z1 + 5.5 * z2)).reshape(-1, 1)
    )
    assert covar_effect.jac_g(np.column_stack((z1, z2)))[1] == approx(
        (z2 * np.exp(2.4 * z1 + 5.5 * z2)).reshape(-1, 1)
    )

    assert covar_effect.g(np.ones(covar_effect.nb_coef)).shape == ()
    assert covar_effect.g(np.ones((1, covar_effect.nb_coef))).shape == (1, 1)
    assert covar_effect.g(np.ones((10, covar_effect.nb_coef))).shape == (10, 1)

    assert covar_effect.jac_g(np.ones(covar_effect.nb_coef)).shape == (
        covar_effect.nb_coef,
    )
    assert covar_effect.jac_g(np.ones(covar_effect.nb_coef)).shape == (
        covar_effect.nb_coef,
    )
    assert covar_effect.jac_g(np.ones((1, covar_effect.nb_coef))).shape == (
        covar_effect.nb_coef,
        1,
        1,
    )
    assert covar_effect.jac_g(np.ones((10, covar_effect.nb_coef))).shape == (
        covar_effect.nb_coef,
        10,
        1,
    )


def test_rvs(regression, covar, rvs_size):
    assert regression.rvs(rvs_size, covar).shape == rvs_expected_shape(
        rvs_size, covar=covar
    )
    assert all(
        arr.shape == rvs_expected_shape(rvs_size, covar=covar)
        for arr in regression.rvs(rvs_size, covar, return_event=True)
    )
    assert all(
        arr.shape == rvs_expected_shape(rvs_size, covar=covar)
        for arr in regression.rvs(rvs_size, covar, return_entry=True)
    )
    assert all(
        arr.shape == rvs_expected_shape(rvs_size, covar=covar)
        for arr in regression.rvs(
            rvs_size,
            covar,
            return_event=True,
            return_entry=True,
        )
    )


def test_sf(regression, time, covar):
    assert regression.sf(time, covar).shape == expected_shape(time=time, covar=covar)


def test_hf(regression, time, covar):
    assert regression.hf(time, covar).shape == expected_shape(time=time, covar=covar)


def test_chf(regression, time, covar):
    assert regression.chf(time, covar).shape == expected_shape(time=time, covar=covar)


def test_cdf(regression, time, covar):
    assert regression.cdf(time, covar).shape == expected_shape(time=time, covar=covar)


def test_pdf(regression, time, covar):
    assert regression.pdf(time, covar).shape == expected_shape(time=time, covar=covar)


def test_ppf(regression, probability, covar):
    assert regression.ppf(probability, covar).shape == expected_shape(
        time=probability, covar=covar
    )


def test_ichf(regression, probability, covar):
    assert regression.ichf(probability, covar).shape == expected_shape(
        time=probability, covar=covar
    )


def test_isf(regression, probability, covar):
    assert regression.isf(probability, covar).shape == expected_shape(
        time=probability, covar=covar
    )
    assert regression.isf(np.full(probability.shape, 0.5), covar) == approx(
        np.broadcast_to(
            regression.median(covar), expected_shape(time=probability, covar=covar)
        )
    )


def test_dhf(regression, time, covar):
    assert regression.dhf(time, covar).shape == expected_shape(time=time, covar=covar)


def test_jac_sf(regression, time, covar):
    assert regression.jac_sf(time, covar).shape == (
        regression.nb_params,
    ) + expected_shape(time=time, covar=covar)


def test_jac_hf(regression, time, covar):
    assert regression.jac_hf(time, covar).shape == (
        regression.nb_params,
    ) + expected_shape(time=time, covar=covar)


def test_jac_chf(regression, time, covar):
    assert regression.jac_chf(time, covar).shape == (
        regression.nb_params,
    ) + expected_shape(time=time, covar=covar)


def test_jac_cdf(regression, time, covar):
    assert regression.jac_cdf(time, covar).shape == (
        regression.nb_params,
    ) + expected_shape(time=time, covar=covar)


def test_jac_pdf(regression, time, covar):
    assert regression.jac_pdf(time, covar).shape == (
        regression.nb_params,
    ) + expected_shape(time=time, covar=covar)


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
        model_args=covar_data,
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
    )
    weibull_pph = ParametricProportionalHazard(Weibull()).fit(
        insulator_string_data["time"],
        model_args=covar_data,
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
    )

    assert weibull_pph.baseline.params == approx(weibull_aft.baseline.params, rel=1e-3)
    assert weibull_pph.covar_effect.params == approx(
        -weibull_aft.baseline.params[0] * weibull_aft.covar_effect.params,
        rel=1e-3,
    )


def test_cox_params_eq(insulator_string_data):
    # From manual experiment and comparison to lifelines results
    insulator_data_cox_params = np.array([5.08787802, -2.98553117, 4.51758019])
    timeline_head = np.array([0.0,
                                     1.1,
                                     2.6,
                                     3.0,
                                     3.1,
                                     3.2,
                                     3.3,
                                     4.0,
                                     4.6,
                                     4.7,
                                     4.9,
                                     5.1,
                                     5.3,
                                     5.4,
                                     6.0,
                                     6.1,
                                     6.4,
                                     6.5,
                                     6.6,
                                     6.9], dtype=np.float64)
    sf_head = np.array([[1.        , 1.        ],
                               [0.99997434, 0.99985195],
                               [0.99994812, 0.99970062],
                               [0.99992171, 0.99954827],
                               [0.99986881, 0.99924318],
                               [0.99984205, 0.99908886],
                               [0.99981526, 0.99893439],
                               [0.99978816, 0.99877815],
                               [0.99976071, 0.9986199 ],
                               [0.99973319, 0.99846125],
                               [0.99970561, 0.9983023 ],
                               [0.99967792, 0.99814277],
                               [0.99959458, 0.99766265],
                               [0.9995667 , 0.9975021 ],
                               [0.99951059, 0.99717901],
                               [0.99948252, 0.99701738],
                               [0.99939808, 0.9965314 ],
                               [0.99936975, 0.9963684 ],
                               [0.99931304, 0.99604217],
                               [0.99925581, 0.99571303]], dtype=np.float64)

    re_model = SemiParametricProportionalHazard()
    covar = np.column_stack(
            (
                insulator_string_data["pHCl"],
                insulator_string_data["pH2SO4"],
                insulator_string_data["HNO3"],
            )
        )
    re_model.fit(
        time=insulator_string_data["time"],
        covar=covar,
        event=insulator_string_data["event"],
    )
    sf_relife = re_model.sf(
        covar=covar[:2, :], se=False
    )

    assert re_model.params == approx(insulator_data_cox_params, rel=1e-3)
    assert sf_relife[0][:20] == approx(timeline_head, rel=1e-3)
    assert np.transpose(sf_relife[1][:, :20]) == approx(sf_head, rel=1e-3)
