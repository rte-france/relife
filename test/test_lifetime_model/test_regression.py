import numpy as np
from pytest import approx
from scipy.stats import boxcox, zscore

from relife.lifetime_model._base import CovarEffect
from relife.lifetime_model import Weibull, AFT, ProportionalHazard


def test_args_names(regression):
    assert regression.args_names == ("covar",)
    assert ProportionalHazard(regression).args_names == ("covar", "covar",)
    assert AFT(regression).args_names == ("covar", "covar",)


def test_covar_effect():
    covar_effect = CovarEffect(0.1)
    assert isinstance(covar_effect.g(np.ones(covar_effect.nb_params)), float)
    assert covar_effect.g(np.ones((10, covar_effect.nb_params))).shape == (10, 1)

    jac_g = covar_effect.jac_g(np.ones(covar_effect.nb_params))
    assert len(jac_g) == covar_effect.nb_params
    assert all(isinstance(jac, float) for jac in jac_g)

    jac_g = covar_effect.jac_g(np.ones((10, covar_effect.nb_params)))
    assert len(jac_g) == covar_effect.nb_params
    assert all(jac.shape == (10, covar_effect.nb_params) for jac in jac_g)

    covar_effect = CovarEffect(0.1, 0.2, 0.3)
    assert isinstance(covar_effect.g(np.ones(covar_effect.nb_params)), float)
    assert covar_effect.g(np.ones((10, covar_effect.nb_params))).shape == (10, 1)

    jac_g = covar_effect.jac_g(np.ones(covar_effect.nb_params))
    assert len(jac_g) == covar_effect.nb_params
    assert all(isinstance(jac, float) for jac in jac_g)

    jac_g = covar_effect.jac_g(np.ones((10, covar_effect.nb_params)))
    assert len(jac_g) == covar_effect.nb_params
    assert all(jac.shape == (10, covar_effect.nb_params) for jac in jac_g)


def test_rvs(regression, covar):
    m,n = 10, 3
    assert regression.rvs(covar(m), seed=21).shape == (m, 1)
    assert regression.rvs((m, 1), covar(m), seed=21).shape == (m, 1)
    assert regression.rvs((m, n), covar(m), seed=21).shape == (m, n)


def test_probability_functions(regression, time, covar, probability):
    m,n = 10, 3

    # covar(m).shape == (m, k)
    assert regression.sf(time(), covar(m)).shape == (m, 1)
    assert regression.sf(regression.median(covar(m)), covar(m)) == approx(np.full((m,1), 0.5), rel=1e-3)
    assert regression.hf(time(), covar(m)).shape == (m, 1)
    assert regression.chf(time(), covar(m)).shape == (m, 1)
    assert regression.cdf(time(), covar(m)).shape == (m, 1)
    assert regression.pdf(time(), covar(m)).shape == (m, 1)
    assert regression.ppf(probability(), covar(m)).shape == (m, 1)
    assert regression.ichf(probability(), covar(m)).shape == (m, 1)
    assert regression.isf(probability(), covar(m)).shape == (m, 1)
    assert regression.isf(0.5, covar(m)) == approx(regression.median(covar(m)))

    # covar(m).shape == (m, k)
    assert regression.sf(time(n), covar(m)).shape == (m, n)
    assert regression.hf(time(n), covar(m)).shape == (m, n)
    assert regression.chf(time(n), covar(m)).shape == (m, n)
    assert regression.cdf(time(n), covar(m)).shape == (m, n)
    assert regression.pdf(time(n), covar(m)).shape == (m, n)
    assert regression.ppf(probability(n,), covar(m),).shape == (m, n)
    assert regression.ichf(probability(n,), covar(m),).shape == (m, n)

    # covar(m).shape == (m, k)
    assert regression.sf(time(m, 1), covar(m)).shape == (m, 1)
    assert regression.hf(time(m, 1), covar(m)).shape == (m, 1)
    assert regression.chf(time(m, 1), covar(m)).shape == (m, 1)
    assert regression.cdf(time(m, 1), covar(m)).shape == (m, 1)
    assert regression.pdf(time(m, 1), covar(m)).shape == (m, 1)
    assert regression.ppf(probability(m, 1), covar(m)).shape == (m, 1)
    assert regression.ichf(probability(m, 1), covar(m)).shape == (m, 1)
    assert regression.isf(probability(m, 1), covar(m)).shape == (m, 1)
    assert regression.isf(np.full((m, 1), 0.5), covar(m)) == approx(regression.median(covar(m)))

    # covar(m).shape == (m, k)
    assert regression.sf(time(m, n), covar(m)).shape == (m, n)
    assert regression.hf(time(m, n), covar(m)).shape == (m, n)
    assert regression.chf(time(m, n), covar(m)).shape == (m, n)
    assert regression.cdf(time(m, n), covar(m)).shape == (m, n)
    assert regression.pdf(time(m, n), covar(m)).shape == (m, n)
    assert regression.ppf(probability(m, n), covar(m)).shape == (m, n)
    assert regression.ichf(probability(m, n), covar(m)).shape == (m, n)
    assert regression.isf(probability(m, n), covar(m)).shape == (m, n)
    assert regression.isf(np.full((m, n), 0.5), covar(m)) == approx(
        np.broadcast_to(regression.median(covar(m)), (m, n))
    )


def test_derivative(regression, time, covar):
    m, n = 3, 10

    assert regression.dhf(time(), covar(m)).shape == (m, 1)
    jac_sf = regression.jac_sf(time(), covar(m))
    jac_hf = regression.jac_hf(time(), covar(m))
    jac_chf = regression.jac_chf(time(), covar(m))
    jac_cdf = regression.jac_cdf(time(), covar(m))
    jac_pdf = regression.jac_pdf(time(), covar(m))
    assert len(jac_sf) == regression.nb_params
    assert len(jac_hf) == regression.nb_params
    assert len(jac_chf) == regression.nb_params
    assert len(jac_cdf) == regression.nb_params
    assert len(jac_pdf) == regression.nb_params
    assert all(jac.shape == (m,1) for jac in jac_sf)
    assert all(jac.shape == (m,1) for jac in jac_hf)
    assert all(jac.shape == (m,1) for jac in jac_chf)
    assert all(jac.shape == (m,1) for jac in jac_cdf)
    assert all(jac.shape == (m,1) for jac in jac_pdf)


    assert regression.dhf(time(n), covar(m)).shape == (m, n)
    assert regression.jac_sf(time(n), covar(m)).shape == (regression.nb_params, m, n)
    assert regression.jac_hf(time(n), covar(m)).shape == (regression.nb_params, m, n)
    assert regression.jac_chf(time(n), covar(m)).shape == (regression.nb_params, m, n)
    assert regression.jac_cdf(time(n), covar(m)).shape == (regression.nb_params, m, n)
    assert regression.jac_pdf(time(n), covar(m)).shape == (regression.nb_params, m, n)

    assert regression.dhf(time(m, 1), covar(m)).shape == (m, 1)
    assert regression.jac_sf(time(m, 1), covar(m)).shape == (regression.nb_params, m, 1)
    assert regression.jac_hf(time(m, 1), covar(m)).shape == (regression.nb_params, m, 1)
    assert regression.jac_chf(time(m, 1), covar(m)).shape == (regression.nb_params, m, 1)
    assert regression.jac_cdf(time(m, 1), covar(m)).shape == (regression.nb_params, m, 1)
    assert regression.jac_pdf(time(m, 1), covar(m)).shape == (regression.nb_params, m, 1)

    assert regression.dhf(time(m, n), covar(m)).shape == (m, n)
    assert regression.jac_sf(time(m, n), covar(m)).shape == (regression.nb_params, m, n)
    assert regression.jac_hf(time(m, n), covar(m)).shape == (regression.nb_params, m, n)
    assert regression.jac_chf(time(m, n), covar(m)).shape == (regression.nb_params, m, n)
    assert regression.jac_cdf(time(m, n), covar(m)).shape == (regression.nb_params, m, n)
    assert regression.jac_pdf(time(m, n), covar(m)).shape == (regression.nb_params, m, n)




def test_ls_integrate(regression, a, b, covar):
    m, n = 2, 3

    # integral_a^b dF(x)
    integration = regression.ls_integrate( np.ones_like, a(), b(), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(regression.cdf(b(), covar(m)) - regression.cdf(a(), covar(m)))
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate( lambda x: x, 0.0, np.inf, covar(m))
    assert integration == approx(regression.mean(covar(m)))

    # integral_a^b dF(x)
    integration = regression.ls_integrate( np.ones_like, a(), b(n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(regression.cdf(b(n), covar(m)) - regression.cdf(a(), covar(m)))
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate( lambda x: x, 0.0, np.full((n,), np.inf), covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = regression.ls_integrate( np.ones_like, a(n), b(), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(regression.cdf(b(), covar(m)) - regression.cdf(a(n), covar(m)))
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate( lambda x: x, np.zeros(n), np.inf, covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = regression.ls_integrate( np.ones_like, a(n), b(n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(regression.cdf(b(n), covar(m)) - regression.cdf(a(n), covar(m)))
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate( lambda x: x, np.zeros(n), np.full((n,), np.inf), covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = regression.ls_integrate( np.ones_like, a(m, 1), b(), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(regression.cdf(b(), covar(m)) - regression.cdf(a(m, 1), covar(m)))
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate( lambda x: x, np.zeros((m,1)), np.inf, covar(m))
    assert integration == approx(np.full((m, 1), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = regression.ls_integrate( np.ones_like, a(), b(m, 1), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(regression.cdf(b(m, 1), covar(m)) - regression.cdf(a(), covar(m)))
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate( lambda x: x, 0.0, np.full((m, 1), np.inf), covar(m))
    assert integration == approx(np.full((m, 1), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = regression.ls_integrate( np.ones_like, a(m, 1), b(m, 1), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(
        regression.cdf(b(m, 1), covar(m)) - regression.cdf(a(m, 1), covar(m))
    )
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate( lambda x: x, np.zeros((m,1)), np.full((m, 1), np.inf), covar(m))
    assert integration == approx(np.full((m, 1), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = regression.ls_integrate( np.ones_like, a(), b(m, n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(regression.cdf(b(m, n), covar(m)) - regression.cdf(a(), covar(m)))
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate( lambda x: x, 0.0, np.full((m, n), np.inf), covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = regression.ls_integrate( np.ones_like, a(m, n), b(), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(regression.cdf(b(), covar(m)) - regression.cdf(a(m, n), covar(m)))
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate( lambda x: x, np.zeros((m, n)), np.inf, covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = regression.ls_integrate( np.ones_like, a(1, n), b(m, 1), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(
        regression.cdf(b(m, 1), covar(m)) - regression.cdf(a(1, n), covar(m))
    )
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate( lambda x: x, np.zeros((m,n)), np.full((m, 1), np.inf), covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = regression.ls_integrate( np.ones_like, a(m, 1), b(1, n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(
        regression.cdf(b(1, n), covar(m)) - regression.cdf(a(m, 1), covar(m))
    )
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate( lambda x: x, np.zeros((m,1)), np.full((1, n), np.inf), covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = regression.ls_integrate( np.ones_like, a(m, n), b(m, n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(
        regression.cdf(b(m, n), covar(m)) - regression.cdf(a(m, n), covar(m))
    )
    # integral_0^inf x*dF(x)
    integration = regression.ls_integrate( lambda x: x, np.zeros((m, n)), np.full((m, n), np.inf), covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))



def test_aft_pph_weibull_eq(insulator_string_data):
    weibull_aft = AFT(Weibull()).fit(
        insulator_string_data[0],
        zscore(np.column_stack([boxcox(v)[0] for v in insulator_string_data[3:]])),
        event=insulator_string_data[1] == 1,
    )
    weibull_pph = ProportionalHazard(Weibull()).fit(
        insulator_string_data[0],
        zscore(np.column_stack([boxcox(v)[0] for v in insulator_string_data[3:]])),
        event=insulator_string_data[1] == 1,
    )

    assert weibull_pph.baseline.params == approx(
        weibull_aft.baseline.params, rel=1e-3
    )
    assert weibull_pph.covar_effect.params == approx(
        -weibull_aft.baseline.shape * weibull_aft.covar_effect.params,
        rel=1e-3,
    )
