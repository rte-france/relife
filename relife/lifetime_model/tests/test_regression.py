import numpy as np
from pytest import approx
from scipy.stats import boxcox, zscore

from relife.lifetime_model import AcceleratedFailureTime, ProportionalHazard, Weibull
from relife.lifetime_model.regression import CovarEffect


def test_covar_effect():
    """
    covar : () or (nb_coef,)
    => g : ()
    => jac_g : (nb_coef,)

    covar : (m, nb_coef)
    => g : (m, 1)
    => jac_g : (nb_coef, m, 1)
    """

    covar_effect = CovarEffect(coefficients=(2.4, 5.5))
    z1 = np.array([1, 2, 3])
    z2 = np.array([0.8, 0.7, 0.5])
    assert covar_effect.g(np.column_stack((z1, z2))) == approx(np.exp(2.4 * z1 + 5.5 * z2).reshape(-1, 1))
    assert covar_effect.jac_g(np.column_stack((z1, z2)))[0] == approx((z1 * np.exp(2.4 * z1 + 5.5 * z2)).reshape(-1, 1))
    assert covar_effect.jac_g(np.column_stack((z1, z2)))[1] == approx((z2 * np.exp(2.4 * z1 + 5.5 * z2)).reshape(-1, 1))

    assert covar_effect.g(np.ones(covar_effect.nb_coef)).shape == ()
    assert covar_effect.g(np.ones((1, covar_effect.nb_coef))).shape == (1, 1)
    assert covar_effect.g(np.ones((10, covar_effect.nb_coef))).shape == (10, 1)

    assert covar_effect.jac_g(np.ones(covar_effect.nb_coef), asarray=True).shape == (covar_effect.nb_coef,)
    assert covar_effect.jac_g(np.ones(covar_effect.nb_coef), asarray=True).shape == (covar_effect.nb_coef,)
    assert covar_effect.jac_g(np.ones((1, covar_effect.nb_coef)), asarray=True).shape == (covar_effect.nb_coef, 1, 1)
    assert covar_effect.jac_g(np.ones((10, covar_effect.nb_coef)), asarray=True).shape == (covar_effect.nb_coef, 10, 1)


def test_rvs(regression, rvs_size, covar, expected_out_shape):
    assert regression.rvs(rvs_size, covar).shape == expected_out_shape(covar=covar, size=rvs_size)
    assert all(
        arr.shape == expected_out_shape(covar=covar, size=rvs_size)
        for arr in regression.rvs(rvs_size, covar, return_event=True)
    )
    assert all(
        arr.shape == expected_out_shape(covar=covar, size=rvs_size)
        for arr in regression.rvs(rvs_size, covar, return_entry=True)
    )
    assert all(
        arr.shape == expected_out_shape(covar=covar, size=rvs_size)
        for arr in regression.rvs(rvs_size, covar, return_event=True, return_entry=True)
    )


def test_sf(regression, time, covar, expected_out_shape):
    assert regression.sf(time, covar).shape == expected_out_shape(time=time, covar=covar)


def test_hf(regression, time, covar, expected_out_shape):
    assert regression.hf(time, covar).shape == expected_out_shape(time=time, covar=covar)


def test_chf(regression, time, covar, expected_out_shape):
    assert regression.chf(time, covar).shape == expected_out_shape(time=time, covar=covar)


def test_cdf(regression, time, covar, expected_out_shape):
    assert regression.cdf(time, covar).shape == expected_out_shape(time=time, covar=covar)


def test_pdf(regression, time, covar, expected_out_shape):
    assert regression.pdf(time, covar).shape == expected_out_shape(time=time, covar=covar)


def test_ppf(regression, probability, covar, expected_out_shape):
    assert regression.ppf(probability, covar).shape == expected_out_shape(time=probability, covar=covar)


def test_ichf(regression, probability, covar, expected_out_shape):
    assert regression.ichf(probability, covar).shape == expected_out_shape(time=probability, covar=covar)


def test_isf(regression, probability, covar, expected_out_shape):
    assert regression.isf(probability, covar).shape == expected_out_shape(time=probability, covar=covar)
    assert regression.isf(np.full(probability.shape, 0.5), covar) == approx(
        np.broadcast_to(regression.median(covar), expected_out_shape(time=probability, covar=covar))
    )


def test_dhf(regression, time, covar, expected_out_shape):
    assert regression.dhf(time, covar).shape == expected_out_shape(time=time, covar=covar)


def test_jac_sf(regression, time, covar, expected_out_shape):
    assert regression.jac_sf(time, covar, asarray=True).shape == (regression.nb_params,) + expected_out_shape(
        time=time, covar=covar
    )


def test_jac_hf(regression, time, covar, expected_out_shape):
    assert regression.jac_hf(time, covar, asarray=True).shape == (regression.nb_params,) + expected_out_shape(
        time=time, covar=covar
    )


def test_jac_chf(regression, time, covar, expected_out_shape):
    assert regression.jac_chf(time, covar, asarray=True).shape == (regression.nb_params,) + expected_out_shape(
        time=time, covar=covar
    )


def test_jac_cdf(regression, time, covar, expected_out_shape):
    assert regression.jac_cdf(time, covar, asarray=True).shape == (regression.nb_params,) + expected_out_shape(
        time=time, covar=covar
    )


def test_jac_pdf(regression, time, covar, expected_out_shape):
    assert regression.jac_pdf(time, covar, asarray=True).shape == (regression.nb_params,) + expected_out_shape(
        time=time, covar=covar
    )


def test_ls_integrate(regression, integration_bound_a, integration_bound_b, covar, expected_out_shape):
    # integral_a^b dF(x)
    integration = regression.ls_integrate(np.ones_like, integration_bound_a, integration_bound_b, covar, deg=100)
    assert integration.shape == expected_out_shape(a=integration_bound_a, b=integration_bound_b, covar=covar)
    assert integration == approx(
        regression.cdf(integration_bound_b, covar) - regression.cdf(integration_bound_a, covar)
    )
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate(
        lambda x: x, np.zeros_like(integration_bound_a), np.full_like(integration_bound_b, np.inf), covar, deg=100
    )
    assert integration == approx(
        np.broadcast_to(
            regression.mean(covar),
            expected_out_shape(a=integration_bound_a, b=integration_bound_b, covar=covar),
        ),
        rel=1e-3,
    )


def test_aft_pph_weibull_eq(insulator_string_data):
    covar = zscore(
        np.column_stack(
            (
                boxcox(insulator_string_data["pHCl"])[0],
                boxcox(insulator_string_data["pH2SO4"])[0],
                boxcox(insulator_string_data["HNO3"])[0],
            )
        )
    )
    weibull_aft = AcceleratedFailureTime(Weibull()).fit(
        insulator_string_data["time"],
        covar,
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
    )
    weibull_pph = ProportionalHazard(Weibull()).fit(
        insulator_string_data["time"],
        covar,
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
    )

    assert weibull_pph.baseline.params == approx(weibull_aft.baseline.params, rel=1e-3)
    assert weibull_pph.covar_effect.params == approx(
        -weibull_aft.baseline.shape * weibull_aft.covar_effect.params,
        rel=1e-3,
    )
