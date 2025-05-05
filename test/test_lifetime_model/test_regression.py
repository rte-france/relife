import numpy as np
from numpy.typing import NDArray
from pytest import approx
from scipy.stats import boxcox, zscore

from relife.lifetime_model import Weibull, AcceleratedFailureTime, ProportionalHazard, CovarEffect


def test_args_names(accelerated_failure_time):
    assert accelerated_failure_time.args_names == ("covar",)
    assert ProportionalHazard(accelerated_failure_time).args_names == (
        "covar",
        "covar",
    )
    assert AcceleratedFailureTime(accelerated_failure_time).args_names == (
        "covar",
        "covar",
    )


def test_covar_effect(covar):
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
    assert covar_effect.g(covar(1, covar_effect.nb_coef)).shape == (1, 1)
    assert covar_effect.g(covar(10, covar_effect.nb_coef)).shape == (10, 1)

    assert covar_effect.jac_g(np.ones(covar_effect.nb_coef), asarray=True).shape == (covar_effect.nb_coef,)
    assert covar_effect.jac_g(np.ones(covar_effect.nb_coef), asarray=True).shape == (covar_effect.nb_coef,)
    assert covar_effect.jac_g(covar(1, covar_effect.nb_coef), asarray=True).shape == (covar_effect.nb_coef, 1, 1)
    assert covar_effect.jac_g(covar(10, covar_effect.nb_coef), asarray=True).shape == (covar_effect.nb_coef, 10, 1)


def expected_out_shape(**kwargs: NDArray[np.float64]):
    covar = kwargs.pop("covar", None)
    covar_shape_contrib = ()
    if covar is not None:
        covar: NDArray[np.float64]
        if covar.ndim == 2:
            covar_shape_contrib = (covar.shape[0], 1)
    return np.broadcast_shapes(*tuple((arr.shape for arr in kwargs.values())), covar_shape_contrib)


class TestProportionalHazard:

    def test_rvs(self, proportional_hazard, covar):
        match covar.shape:
            case (_,):
                m, n = 10, 20
                assert proportional_hazard.rvs(covar, seed=21).shape == ()
                assert proportional_hazard.rvs(covar, size=n, seed=21).shape == (n,)
                assert proportional_hazard.rvs(covar, size=(n,), seed=21).shape == (n,)
                assert proportional_hazard.rvs(covar, size=(m, n), seed=21).shape == (m, n)
            case (1, _):
                m, n = 10, 20
                assert proportional_hazard.rvs(covar, seed=21).shape == (1, 1)
                assert proportional_hazard.rvs(covar, size=n, seed=21).shape == (1, n)
                assert proportional_hazard.rvs(covar, size=(n,), seed=21).shape == (1, n)
                assert proportional_hazard.rvs(covar, size=(m, n), seed=21).shape == (m, n)
            case (m, _):
                n = 20
                assert proportional_hazard.rvs(covar, seed=21).shape == (m, 1)
                assert proportional_hazard.rvs(covar, size=n, seed=21).shape == (m, n)
                assert proportional_hazard.rvs(covar, size=(n,), seed=21).shape == (m, n)
                assert proportional_hazard.rvs(covar, size=(m, n), seed=21).shape == (m, n)

    def test_args_names(self, proportional_hazard):
        assert proportional_hazard.args_names == ("covar",)
        assert ProportionalHazard(proportional_hazard).args_names == (
            "covar",
            "covar",
        )

    def test_sf(self, proportional_hazard, time, covar):
        assert proportional_hazard.sf(time, covar).shape == expected_out_shape(time=time, covar=covar)

    def test_hf(self, proportional_hazard, time, covar):
        assert proportional_hazard.hf(time, covar).shape == expected_out_shape(time=time, covar=covar)

    def test_chf(self, proportional_hazard, time, covar):
        assert proportional_hazard.chf(time, covar).shape == expected_out_shape(time=time, covar=covar)

    def test_cdf(self, proportional_hazard, time, covar):
        assert proportional_hazard.cdf(time, covar).shape == expected_out_shape(time=time, covar=covar)

    def test_pdf(self, proportional_hazard, time, covar):
        assert proportional_hazard.pdf(time, covar).shape == expected_out_shape(time=time, covar=covar)

    def test_ppf(self, proportional_hazard, probability, covar):
        assert proportional_hazard.ppf(probability, covar).shape == expected_out_shape(time=probability, covar=covar)

    def test_ichf(self, proportional_hazard, probability, covar):
        assert proportional_hazard.ichf(probability, covar).shape == expected_out_shape(time=probability, covar=covar)

    def test_isf(self, proportional_hazard, probability, covar):
        assert proportional_hazard.isf(probability, covar).shape == expected_out_shape(time=probability, covar=covar)
        assert proportional_hazard.isf(np.full(probability.shape, 0.5), covar) == approx(
            np.broadcast_to(proportional_hazard.median(covar), expected_out_shape(time=probability, covar=covar))
        )

    def test_dhf(self, proportional_hazard, time, covar):
        assert proportional_hazard.dhf(time, covar).shape == expected_out_shape(time=time, covar=covar)

    def test_jac_sf(self, proportional_hazard, time, covar):
        assert proportional_hazard.jac_sf(time, covar, asarray=True).shape == (
            proportional_hazard.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_jac_hf(self, proportional_hazard, time, covar):
        assert proportional_hazard.jac_hf(time, covar, asarray=True).shape == (
            proportional_hazard.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_jac_chf(self, proportional_hazard, time, covar):
        assert proportional_hazard.jac_chf(time, covar, asarray=True).shape == (
            proportional_hazard.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_jac_cdf(self, proportional_hazard, time, covar):
        assert proportional_hazard.jac_cdf(time, covar, asarray=True).shape == (
            proportional_hazard.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_jac_pdf(self, proportional_hazard, time, covar):
        assert proportional_hazard.jac_pdf(time, covar, asarray=True).shape == (
            proportional_hazard.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_ls_integrate(self, proportional_hazard, integration_bound_a, integration_bound_b, covar):
        # integral_a^b dF(x)
        integration = proportional_hazard.ls_integrate(
            np.ones_like, integration_bound_a, integration_bound_b, covar, deg=100
        )
        assert integration.shape == expected_out_shape(a=integration_bound_a, b=integration_bound_b, covar=covar)
        assert integration == approx(
            proportional_hazard.cdf(integration_bound_b, covar) - proportional_hazard.cdf(integration_bound_a, covar)
        )
        # integral_0^inf x*dF(x)
        integration = proportional_hazard.ls_integrate(
            lambda x: x, np.zeros_like(integration_bound_a), np.full_like(integration_bound_b, np.inf), covar, deg=100
        )
        assert integration == approx(
            np.broadcast_to(
                proportional_hazard.mean(covar),
                expected_out_shape(a=integration_bound_a, b=integration_bound_b, covar=covar),
            ),
            rel=1e-3,
        )

    def test_fit(self, proportional_hazard, insulator_string_data):
        proportional_hazard.fit(
            insulator_string_data[0],
            zscore(np.column_stack([boxcox(v)[0] for v in insulator_string_data[3:]])),
            event=insulator_string_data[1] == 1,
        )


class TestAcceleratedFailureTime:

    def test_rvs(self, accelerated_failure_time, covar):
        match covar.shape:
            case (_,):
                m, n = 10, 20
                assert accelerated_failure_time.rvs(covar, seed=21).shape == ()
                assert accelerated_failure_time.rvs(covar, size=n, seed=21).shape == (n,)
                assert accelerated_failure_time.rvs(covar, size=(n,), seed=21).shape == (n,)
                assert accelerated_failure_time.rvs(covar, size=(m, n), seed=21).shape == (m, n)
            case (1, _):
                m, n = 10, 20
                assert accelerated_failure_time.rvs(covar, seed=21).shape == (1, 1)
                assert accelerated_failure_time.rvs(covar, size=n, seed=21).shape == (1, n)
                assert accelerated_failure_time.rvs(covar, size=(n,), seed=21).shape == (1, n)
                assert accelerated_failure_time.rvs(covar, size=(m, n), seed=21).shape == (m, n)
            case (m, _):
                n = 20
                assert accelerated_failure_time.rvs(covar, seed=21).shape == (m, 1)
                assert accelerated_failure_time.rvs(covar, size=n, seed=21).shape == (m, n)
                assert accelerated_failure_time.rvs(covar, size=(n,), seed=21).shape == (m, n)
                assert accelerated_failure_time.rvs(covar, size=(m, n), seed=21).shape == (m, n)

    def test_args_names(self, accelerated_failure_time):
        assert accelerated_failure_time.args_names == ("covar",)
        assert ProportionalHazard(accelerated_failure_time).args_names == (
            "covar",
            "covar",
        )

    def test_sf(self, accelerated_failure_time, time, covar):
        assert accelerated_failure_time.sf(time, covar).shape == expected_out_shape(time=time, covar=covar)

    def test_hf(self, accelerated_failure_time, time, covar):
        assert accelerated_failure_time.hf(time, covar).shape == expected_out_shape(time=time, covar=covar)

    def test_chf(self, accelerated_failure_time, time, covar):
        assert accelerated_failure_time.chf(time, covar).shape == expected_out_shape(time=time, covar=covar)

    def test_cdf(self, accelerated_failure_time, time, covar):
        assert accelerated_failure_time.cdf(time, covar).shape == expected_out_shape(time=time, covar=covar)

    def test_pdf(self, accelerated_failure_time, time, covar):
        assert accelerated_failure_time.pdf(time, covar).shape == expected_out_shape(time=time, covar=covar)

    def test_ppf(self, accelerated_failure_time, probability, covar):
        assert accelerated_failure_time.ppf(probability, covar).shape == expected_out_shape(
            time=probability, covar=covar
        )

    def test_ichf(self, accelerated_failure_time, probability, covar):
        assert accelerated_failure_time.ichf(probability, covar).shape == expected_out_shape(
            time=probability, covar=covar
        )

    def test_isf(self, accelerated_failure_time, probability, covar):
        assert accelerated_failure_time.isf(probability, covar).shape == expected_out_shape(
            time=probability, covar=covar
        )
        assert accelerated_failure_time.isf(np.full(probability.shape, 0.5), covar) == approx(
            np.broadcast_to(accelerated_failure_time.median(covar), expected_out_shape(time=probability, covar=covar))
        )

    def test_dhf(self, accelerated_failure_time, time, covar):
        assert accelerated_failure_time.dhf(time, covar).shape == expected_out_shape(time=time, covar=covar)

    def test_jac_sf(self, accelerated_failure_time, time, covar):
        assert accelerated_failure_time.jac_sf(time, covar, asarray=True).shape == (
            accelerated_failure_time.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_jac_hf(self, accelerated_failure_time, time, covar):
        assert accelerated_failure_time.jac_hf(time, covar, asarray=True).shape == (
            accelerated_failure_time.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_jac_chf(self, accelerated_failure_time, time, covar):
        assert accelerated_failure_time.jac_chf(time, covar, asarray=True).shape == (
            accelerated_failure_time.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_jac_cdf(self, accelerated_failure_time, time, covar):
        assert accelerated_failure_time.jac_cdf(time, covar, asarray=True).shape == (
            accelerated_failure_time.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_jac_pdf(self, accelerated_failure_time, time, covar):
        assert accelerated_failure_time.jac_pdf(time, covar, asarray=True).shape == (
            accelerated_failure_time.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_ls_integrate(self, accelerated_failure_time, integration_bound_a, integration_bound_b, covar):
        # integral_a^b dF(x)
        integration = accelerated_failure_time.ls_integrate(
            np.ones_like, integration_bound_a, integration_bound_b, covar, deg=100
        )
        assert integration.shape == expected_out_shape(a=integration_bound_a, b=integration_bound_b, covar=covar)
        assert integration == approx(
            accelerated_failure_time.cdf(integration_bound_b, covar)
            - accelerated_failure_time.cdf(integration_bound_a, covar)
        )
        # integral_0^inf x*dF(x)
        integration = accelerated_failure_time.ls_integrate(
            lambda x: x, np.zeros_like(integration_bound_a), np.full_like(integration_bound_b, np.inf), covar, deg=100
        )
        assert integration == approx(
            np.broadcast_to(
                accelerated_failure_time.mean(covar),
                expected_out_shape(a=integration_bound_a, b=integration_bound_b, covar=covar),
            ),
            rel=1e-3,
        )

    def test_fit(self, accelerated_failure_time, insulator_string_data):
        accelerated_failure_time.fit(
            insulator_string_data[0],
            zscore(np.column_stack([boxcox(v)[0] for v in insulator_string_data[3:]])),
            event=insulator_string_data[1] == 1,
        )


# # @pytest.mark.xfail
def test_aft_pph_weibull_eq(insulator_string_data):

    weibull_aft = AcceleratedFailureTime(Weibull()).fit(
        insulator_string_data[0],
        zscore(np.column_stack([boxcox(v)[0] for v in insulator_string_data[3:]])),
        event=insulator_string_data[1] == 1,
    )
    weibull_pph = ProportionalHazard(Weibull()).fit(
        insulator_string_data[0],
        zscore(np.column_stack([boxcox(v)[0] for v in insulator_string_data[3:]])),
        event=insulator_string_data[1] == 1,
    )

    assert weibull_pph.baseline.params == approx(weibull_aft.baseline.params, rel=1e-3)
    assert weibull_pph.covar_effect.params == approx(
        -weibull_aft.baseline.shape * weibull_aft.covar_effect.params,
        rel=1e-3,
    )
