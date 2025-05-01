import numpy as np
import pytest
from pytest import approx
from scipy.stats import boxcox, zscore

from relife.lifetime_model._base import CovarEffect
from relife.lifetime_model import Weibull, AcceleratedFailureTime, ProportionalHazard


# def test_args_names(regression):
#     assert regression.args_names == ("covar",)
#     assert ProportionalHazard(regression).args_names == ("covar", "covar",)
#     assert AcceleratedFailureTime(regression).args_names == ("covar", "covar",)
#
#
#
def test_covar_effect(covar):
    """
    covar : () or (nb_coef,)
    => g : ()
    => jac_g : (nb_coef,)

    covar : (m, nb_coef)
    => g : (m, 1)
    => jac_g : (nb_coef, m, 1)
    """

    covar_effect = CovarEffect(2.4, 5.5)
    z1 = np.array([1, 2, 3])
    z2 = np.array([0.8, 0.7, 0.5])
    assert covar_effect.g(np.column_stack((z1, z2))) == approx(np.exp(2.4 * z1 + 5.5 * z2).reshape(-1, 1))
    assert covar_effect.jac_g(np.column_stack((z1, z2)))[0] ==  approx((z1*np.exp(2.4 * z1 + 5.5 * z2)).reshape(-1, 1))
    assert covar_effect.jac_g(np.column_stack((z1, z2)))[1] == approx((z2*np.exp(2.4 * z1 + 5.5 * z2)).reshape(-1, 1))

    assert covar_effect.g(np.ones(covar_effect.nb_coef)).shape == ()
    assert covar_effect.g(covar(1, covar_effect.nb_coef)).shape == (1, 1)
    assert covar_effect.g(covar(10, covar_effect.nb_coef)).shape == (10, 1)

    assert covar_effect.jac_g(np.ones(covar_effect.nb_coef), asarray=True).shape == (covar_effect.nb_coef,)
    assert covar_effect.jac_g(np.ones(covar_effect.nb_coef), asarray=True).shape == (covar_effect.nb_coef,)
    assert covar_effect.jac_g(covar(1, covar_effect.nb_coef), asarray=True).shape == (covar_effect.nb_coef, 1, 1)
    assert covar_effect.jac_g(covar(10, covar_effect.nb_coef), asarray=True).shape == (covar_effect.nb_coef, 10, 1)


# def test_rvs(regression, covar):
#     m,n = 10, 3
#     assert regression.rvs(covar(m, regression.nb_coef), seed=21).shape == (m, 1)
#     assert regression.rvs(covar(m, regression.nb_coef), shape=(m,1), seed=21).shape == (m, 1)
#     assert regression.rvs(covar(m, regression.nb_coef), shape=(m,n), seed=21).shape == (m, n)
#
# def test_sf(regression, time, covar):
#     assert regression.sf(time, covar(regression.nb_coef,)).shape == time.shape
#     m = 10 if time.ndim <=1 else time.shape[0]
#     n = time.size if time.ndim <=1 else time.shape[1]
#     assert regression.sf(time, covar(m, regression.nb_coef)).shape == (m, n)
#
# def test_hf(regression, time, covar):
#     assert regression.hf(time, covar(regression.nb_coef,)).shape == time.shape
#     m = 10 if time.ndim <=1 else time.shape[0]
#     n = time.size if time.ndim <=1 else time.shape[1]
#     assert regression.hf(time, covar(m, regression.nb_coef)).shape == (m, n)
#
# def test_chf(regression, time, covar):
#     assert regression.chf(time, covar(regression.nb_coef,)).shape == time.shape
#     m = 10 if time.ndim <=1 else time.shape[0]
#     n = time.size if time.ndim <=1 else time.shape[1]
#     assert regression.chf(time, covar(m, regression.nb_coef)).shape == (m, n)
#
# def test_cdf(regression, time, covar):
#     assert regression.cdf(time, covar(regression.nb_coef,)).shape == time.shape
#     m = 10 if time.ndim <=1 else time.shape[0]
#     n = time.size if time.ndim <=1 else time.shape[1]
#     assert regression.cdf(time, covar(m, regression.nb_coef)).shape == (m, n)
#
# def test_pdf(regression, time, covar):
#     assert regression.pdf(time, covar(regression.nb_coef,)).shape == time.shape
#     m = 10 if time.ndim <=1 else time.shape[0]
#     n = time.size if time.ndim <=1 else time.shape[1]
#     assert regression.pdf(time, covar(m, regression.nb_coef)).shape == (m, n)
#
# def test_ppf(regression, probability, covar):
#     assert regression.ppf(probability, covar(regression.nb_coef,)).shape == probability.shape
#     m = 1 if probability.ndim <=1 else probability.shape[0]
#     n = probability.size if probability.ndim <=1 else probability.shape[1]
#     assert regression.ppf(probability, covar(m, regression.nb_coef)).shape == (m, n)
#
# def test_ichf(regression, probability, covar):
#     assert regression.ichf(probability, covar(regression.nb_coef,)).shape == probability.shape
#     m = 1 if probability.ndim <=1 else probability.shape[0]
#     n = probability.size if probability.ndim <=1 else probability.shape[1]
#     assert regression.ichf(probability, covar(m, regression.nb_coef)).shape == (m, n)
#
# def test_isf(regression, probability, covar):
#     assert regression.isf(probability, covar(regression.nb_coef,)).shape == probability.shape
#     assert regression.isf(np.full(probability.shape, 0.5), covar(regression.nb_coef,)) == approx(
#         np.broadcast_to(regression.median(covar(regression.nb_coef,)), probability.shape)
#     )
#     m = 1 if probability.ndim <=1 else probability.shape[0]
#     n = probability.size if probability.ndim <=1 else probability.shape[1]
#     assert regression.isf(probability, covar(m, regression.nb_coef)).shape == (m, n)
#     assert regression.isf(np.full(probability.shape, 0.5), covar(m, regression.nb_coef)) == approx(
#         np.broadcast_to(regression.median(covar(m, regression.nb_coef)), (m, n))
#     )
#
#
# def test_dhf(regression, time, covar):
#     assert regression.dhf(time, covar(regression.nb_coef,)).shape == time.shape
#     m = 10 if time.ndim <=1 else time.shape[0]
#     n = time.size if time.ndim <=1 else time.shape[1]
#     assert regression.dhf(time, covar(m, regression.nb_coef)).shape == (m, n)
#
# def test_jac_sf(regression, time, covar):
#     assert regression.jac_sf(time, covar(regression.nb_coef,), asarray=True).shape == (regression.nb_params,) + time.shape
#     m = 10 if time.ndim <=1 else time.shape[0]
#     n = time.size if time.ndim <=1 else time.shape[1]
#     assert regression.jac_sf(time, covar(m, regression.nb_coef), asarray=True).shape == (regression.nb_params, m, n)
#
# def test_jac_hf(regression, time, covar):
#     assert regression.jac_hf(time, covar(regression.nb_coef,), asarray=True).shape == (regression.nb_params,) + time.shape
#     m = 10 if time.ndim <=1 else time.shape[0]
#     n = time.size if time.ndim <=1 else time.shape[1]
#     assert regression.jac_hf(time, covar(m, regression.nb_coef), asarray=True).shape == (regression.nb_params, m, n)
#
# def test_jac_chf(regression, time, covar):
#     assert regression.jac_chf(time, covar(regression.nb_coef,), asarray=True).shape == (regression.nb_params,) + time.shape
#     m = 10 if time.ndim <=1 else time.shape[0]
#     n = time.size if time.ndim <=1 else time.shape[1]
#     assert regression.jac_chf(time, covar(m, regression.nb_coef), asarray=True).shape == (regression.nb_params, m, n)
#
# def test_jac_cdf(regression, time, covar):
#     assert regression.jac_cdf(time, covar(regression.nb_coef,), asarray=True).shape == (regression.nb_params,) + time.shape
#     m = 10 if time.ndim <=1 else time.shape[0]
#     n = time.size if time.ndim <=1 else time.shape[1]
#     assert regression.jac_cdf(time, covar(m, regression.nb_coef), asarray=True).shape == (regression.nb_params, m, n)
#
#
# def test_jac_pdf(regression, time, covar):
#     assert regression.jac_pdf(time, covar(regression.nb_coef,), asarray=True).shape == (regression.nb_params,) + time.shape
#     m = 10 if time.ndim <=1 else time.shape[0]
#     n = time.size if time.ndim <=1 else time.shape[1]
#     assert regression.jac_pdf(time, covar(m, regression.nb_coef), asarray=True).shape == (regression.nb_params, m, n)


# def test_ls_integrate(regression, a, b, covar):
#     m, n = 2, 3
#
#     # integral_a^b dF(x)
#     integration = regression.ls_integrate( np.ones_like, a(), b(), covar(m, regression.nb_coef), deg=100)
#     assert integration.shape == (m, 1)
#     assert integration == approx(regression.cdf(b(), covar(m, regression.nb_coef)) - regression.cdf(a(), covar(m, regression.nb_coef)))
#     # integral_0^inf x*dF(x)
#     integration = regression.ls_integrate( lambda x: x, 0.0, np.inf, covar(m, regression.nb_coef), deg=100)
#     assert integration == approx(regression.mean(covar(m, regression.nb_coef)))
#
#     # integral_a^b dF(x)
#     integration = regression.ls_integrate( np.ones_like, a(), b(n), covar(m, regression.nb_coef), deg=100)
#     assert integration.shape == (m, n)
#     assert integration == approx(regression.cdf(b(n), covar(m, regression.nb_coef)) - regression.cdf(a(), covar(m, regression.nb_coef)))
#     # integral_0^inf x*dF(x)
#     integration = regression.ls_integrate( lambda x: x, 0.0, np.full((n,), np.inf), covar(m, regression.nb_coef), deg=100)
#     assert integration == approx(np.full((m, n), regression.mean(covar(m, regression.nb_coef))))
#
#     # integral_a^b dF(x)
#     integration = regression.ls_integrate( np.ones_like, a(n), b(), covar(m, regression.nb_coef), deg=100)
#     assert integration.shape == (m, n)
#     assert integration == approx(regression.cdf(b(), covar(m, regression.nb_coef)) - regression.cdf(a(n), covar(m, regression.nb_coef)))
#     # integral_0^inf x*dF(x)
#     integration = regression.ls_integrate( lambda x: x, np.zeros(n), np.inf, covar(m, regression.nb_coef), deg=100)
#     assert integration == approx(np.full((m, n), regression.mean(covar(m, regression.nb_coef))))
#
#     # integral_a^b dF(x)
#     integration = regression.ls_integrate( np.ones_like, a(n), b(n), covar(m, regression.nb_coef), deg=100)
#     assert integration.shape == (m, n)
#     assert integration == approx(regression.cdf(b(n), covar(m, regression.nb_coef)) - regression.cdf(a(n), covar(m, regression.nb_coef)))
#     # integral_0^inf x*dF(x)
#     integration = regression.ls_integrate( lambda x: x, np.zeros(n), np.full((n,), np.inf), covar(m, regression.nb_coef), deg=100)
#     assert integration == approx(np.full((m, n), regression.mean(covar(m, regression.nb_coef))))
#
#     # integral_a^b dF(x)
#     integration = regression.ls_integrate( np.ones_like, a(m, 1), b(), covar(m, regression.nb_coef), deg=100)
#     assert integration.shape == (m, 1)
#     assert integration == approx(regression.cdf(b(), covar(m, regression.nb_coef)) - regression.cdf(a(m, 1), covar(m, regression.nb_coef)))
#     # integral_0^inf x*dF(x)
#     integration = regression.ls_integrate( lambda x: x, np.zeros((m,1)), np.inf, covar(m, regression.nb_coef), deg=100)
#     assert integration == approx(np.full((m, 1), regression.mean(covar(m, regression.nb_coef))))
#
#     # integral_a^b dF(x)
#     integration = regression.ls_integrate( np.ones_like, a(), b(m, 1), covar(m, regression.nb_coef), deg=100)
#     assert integration.shape == (m, 1)
#     assert integration == approx(regression.cdf(b(m, 1), covar(m, regression.nb_coef)) - regression.cdf(a(), covar(m, regression.nb_coef)))
#     # integral_0^inf x*dF(x)
#     integration = regression.ls_integrate( lambda x: x, 0.0, np.full((m, 1), np.inf), covar(m, regression.nb_coef), deg=100)
#     assert integration == approx(np.full((m, 1), regression.mean(covar(m, regression.nb_coef))))
#
#     # integral_a^b dF(x)
#     integration = regression.ls_integrate( np.ones_like, a(m, 1), b(m, 1), covar(m, regression.nb_coef), deg=100)
#     assert integration.shape == (m, 1)
#     assert integration == approx(
#         regression.cdf(b(m, 1), covar(m, regression.nb_coef)) - regression.cdf(a(m, 1), covar(m, regression.nb_coef))
#     )
#     # integral_0^inf x*dF(x)
#     integration = regression.ls_integrate( lambda x: x, np.zeros((m,1)), np.full((m, 1), np.inf), covar(m, regression.nb_coef), deg=100)
#     assert integration == approx(np.full((m, 1), regression.mean(covar(m, regression.nb_coef))))
#
#     # integral_a^b dF(x)
#     integration = regression.ls_integrate( np.ones_like, a(), b(m, n), covar(m, regression.nb_coef), deg=100)
#     assert integration.shape == (m, n)
#     assert integration == approx(regression.cdf(b(m, n), covar(m, regression.nb_coef)) - regression.cdf(a(), covar(m, regression.nb_coef)))
#     # integral_0^inf x*dF(x)
#     integration = regression.ls_integrate( lambda x: x, 0.0, np.full((m, n), np.inf), covar(m, regression.nb_coef), deg=100)
#     assert integration == approx(np.full((m, n), regression.mean(covar(m, regression.nb_coef))))
#
#     # integral_a^b dF(x)
#     integration = regression.ls_integrate( np.ones_like, a(m, n), b(), covar(m, regression.nb_coef), deg=100)
#     assert integration.shape == (m, n)
#     assert integration == approx(regression.cdf(b(), covar(m, regression.nb_coef)) - regression.cdf(a(m, n), covar(m, regression.nb_coef)))
#     # integral_0^inf x*dF(x)
#     integration = regression.ls_integrate( lambda x: x, np.zeros((m, n)), np.inf, covar(m, regression.nb_coef), deg=100)
#     assert integration == approx(np.full((m, n), regression.mean(covar(m, regression.nb_coef))))
#
#     # integral_a^b dF(x)
#     integration = regression.ls_integrate(np.ones_like, a(1, n), b(m, 1), covar(m, regression.nb_coef), deg=100)
#     assert integration.shape == (m, n)
#     assert integration == approx(
#         regression.cdf(b(m, 1), covar(m, regression.nb_coef)) - regression.cdf(a(1, n), covar(m, regression.nb_coef))
#     )
#     # integral_0^inf x*dF(x)
#     integration = regression.ls_integrate( lambda x: x, np.zeros((m,n)), np.full((m, 1), np.inf), covar(m, regression.nb_coef), deg=100)
#     assert integration == approx(np.full((m, n), regression.mean(covar(m, regression.nb_coef))))
#
#     # integral_a^b dF(x)
#     integration = regression.ls_integrate( np.ones_like, a(m, 1), b(1, n), covar(m, regression.nb_coef), deg=100)
#     assert integration.shape == (m, n)
#     assert integration == approx(
#         regression.cdf(b(1, n), covar(m, regression.nb_coef)) - regression.cdf(a(m, 1), covar(m, regression.nb_coef))
#     )
#     # integral_0^inf x*dF(x)
#     integration = regression.ls_integrate( lambda x: x, np.zeros((m,1)), np.full((1, n), np.inf), covar(m, regression.nb_coef), deg=100)
#     assert integration == approx(np.full((m, n), regression.mean(covar(m, regression.nb_coef))))
#
#     # integral_a^b dF(x)
#     integration = regression.ls_integrate( np.ones_like, a(m, n), b(m, n), covar(m, regression.nb_coef), deg=100)
#     assert integration.shape == (m, n)
#     assert integration == approx(
#         regression.cdf(b(m, n), covar(m, regression.nb_coef)) - regression.cdf(a(m, n), covar(m, regression.nb_coef))
#     )
#     # integral_0^inf x*dF(x)
#     integration = regression.ls_integrate( lambda x: x, np.zeros((m, n)), np.full((m, n), np.inf), covar(m, regression.nb_coef), deg=100)
#     assert integration == approx(np.full((m, n), regression.mean(covar(m, regression.nb_coef))))


# # @pytest.mark.xfail
def test_fit(regression, insulator_string_data):
    regression.fit(
        insulator_string_data[0],
        zscore(np.column_stack([boxcox(v)[0] for v in insulator_string_data[3:]])),
        event=insulator_string_data[1] == 1,
    )

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

    assert weibull_pph.baseline.params == approx(
        weibull_aft.baseline.params, rel=1e-3
    )
    assert weibull_pph.covar_effect.params == approx(
        -weibull_aft.baseline.shape * weibull_aft.covar_effect.params,
        rel=1e-3,
    )
