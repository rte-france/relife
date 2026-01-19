# pyright: basic

import numpy as np
from pytest import approx
from scipy.stats import boxcox, zscore

from relife.lifetime_model import AcceleratedFailureTime, ProportionalHazard, Weibull
from relife.lifetime_model._regression import CovarEffect


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


def rvs_expected_shape(size, nb_assets=None, **kwargs):
    out_shape = expected_shape(**kwargs)
    if nb_assets is not None:
        return np.broadcast_shapes(out_shape, (nb_assets, size))
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


def test_rvs(regression, covar, rvs_size, rvs_nb_assets):
    assert regression.rvs(rvs_size, covar, nb_assets=rvs_nb_assets).shape == rvs_expected_shape(
        rvs_size, nb_assets=rvs_nb_assets, covar=covar
    )
    assert all(
        arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, covar=covar)
        for arr in regression.rvs(rvs_size, covar, nb_assets=rvs_nb_assets, return_event=True)
    )
    assert all(
        arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, covar=covar)
        for arr in regression.rvs(rvs_size, covar, nb_assets=rvs_nb_assets, return_entry=True)
    )
    assert all(
        arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, covar=covar)
        for arr in regression.rvs(
            rvs_size,
            covar,
            nb_assets=rvs_nb_assets,
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
    assert regression.ppf(probability, covar).shape == expected_shape(time=probability, covar=covar)


def test_ichf(regression, probability, covar):
    assert regression.ichf(probability, covar).shape == expected_shape(time=probability, covar=covar)


def test_isf(regression, probability, covar):
    assert regression.isf(probability, covar).shape == expected_shape(time=probability, covar=covar)
    assert regression.isf(np.full(probability.shape, 0.5), covar) == approx(
        np.broadcast_to(regression.median(covar), expected_shape(time=probability, covar=covar))
    )


def test_dhf(regression, time, covar):
    assert regression.dhf(time, covar).shape == expected_shape(time=time, covar=covar)


def test_jac_sf(regression, time, covar):
    assert regression.jac_sf(time, covar, asarray=True).shape == (regression.nb_params,) + expected_shape(
        time=time, covar=covar
    )


def test_jac_hf(regression, time, covar):
    assert regression.jac_hf(time, covar, asarray=True).shape == (regression.nb_params,) + expected_shape(
        time=time, covar=covar
    )


def test_jac_chf(regression, time, covar):
    assert regression.jac_chf(time, covar, asarray=True).shape == (regression.nb_params,) + expected_shape(
        time=time, covar=covar
    )


def test_jac_cdf(regression, time, covar):
    assert regression.jac_cdf(time, covar, asarray=True).shape == (regression.nb_params,) + expected_shape(
        time=time, covar=covar
    )


def test_jac_pdf(regression, time, covar):
    assert regression.jac_pdf(time, covar, asarray=True).shape == (regression.nb_params,) + expected_shape(
        time=time, covar=covar
    )


def test_ls_integrate(regression, integration_bound_a, integration_bound_b, covar):
    # integral_a^b dF(x)
    integration = regression.ls_integrate(np.ones_like, integration_bound_a, integration_bound_b, covar, deg=100)
    assert integration.shape == expected_shape(a=integration_bound_a, b=integration_bound_b, covar=covar)
    assert integration == approx(
        regression.cdf(integration_bound_b, covar) - regression.cdf(integration_bound_a, covar)
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
    weibull_aft = AcceleratedFailureTime(Weibull()).fit(
        insulator_string_data["time"],
        model_args=covar_data,
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
    )
    weibull_pph = ProportionalHazard(Weibull()).fit(
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
