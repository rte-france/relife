from pytest import approx
import numpy as np

class TestDistribution:
    def test_args_nb_assets(self, distribution):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.args_nb_assets == 1

    def test_sf(self, distribution, time):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.sf(time).shape == time.shape
        assert frozen_distribution.sf(np.full(time.shape, distribution.median())) == approx(np.full(time.shape, 0.5), rel=1e-3)

    def test_hf(self, distribution, time):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.hf(time).shape == time.shape

    def test_chf(self, distribution, time):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.chf(time).shape == time.shape

    def test_cdf(self, distribution, time):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.cdf(time).shape == time.shape

    def test_pdf(self, distribution, time):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.pdf(time).shape == time.shape

    def test_ppf(self, distribution, probability):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.ppf(probability).shape == probability.shape

    def test_ichf(self, distribution, probability):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.ichf(probability).shape == probability.shape

    def test_isf(self, distribution, probability):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.isf(probability).shape == probability.shape
        assert frozen_distribution.isf(np.full(probability.shape, 0.5)) == approx(
            np.full(probability.shape, distribution.median()))

    def test_moment(self, distribution, time):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.moment(1).shape == ()
        assert frozen_distribution.moment(2).shape == ()
        assert frozen_distribution.mean().shape == ()
        assert frozen_distribution.var().shape == ()
        assert frozen_distribution.median().shape == ()

    def test_dhf(self, distribution, time):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.dhf(time).shape == time.shape

    def test_jac_sf(self, distribution, time):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.jac_sf(time, asarray=True).shape == (frozen_distribution.nb_params,) + time.shape

    def test_jac_hf(self, distribution, time):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.jac_hf(time, asarray=True).shape == (frozen_distribution.nb_params,) + time.shape

    def test_jac_chf(self, distribution, time):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.jac_chf(time, asarray=True).shape == (frozen_distribution.nb_params,) + time.shape

    def test_jac_cdf(self, distribution, time):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.jac_cdf(time, asarray=True).shape == (frozen_distribution.nb_params,) + time.shape

    def test_jac_pdf(self, distribution, time):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.jac_pdf(time, asarray=True).shape == (frozen_distribution.nb_params,) + time.shape



class TestRegression:
    def test_args_nb_assets(self, regression, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef,))
        assert frozen_regression.args_nb_assets == 1
        frozen_regression = regression.freeze(covar(10, regression.nb_coef))
        assert frozen_regression.args_nb_assets == 10

    def test_rvs(self, regression, covar):
        m, n = 10, 3
        frozen_regression = regression.freeze(covar(regression.nb_coef,))
        assert frozen_regression.rvs(seed=21).shape == ()
        assert frozen_regression.rvs(size=(m, 1), seed=21).shape == (m, 1)
        assert frozen_regression.rvs(size=(m, n), seed=21).shape == (m, n)

    def test_sf(self, regression, time, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.sf(time).shape == time.shape
        m = 10 if time.ndim <= 1 else time.shape[0]
        n = time.size if time.ndim <= 1 else time.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.sf(time).shape == (m, n)

    def test_hf(self, regression, time, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.hf(time).shape == time.shape
        m = 10 if time.ndim <= 1 else time.shape[0]
        n = time.size if time.ndim <= 1 else time.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.hf(time).shape == (m, n)

    def test_chf(self, regression, time, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.chf(time).shape == time.shape
        m = 10 if time.ndim <= 1 else time.shape[0]
        n = time.size if time.ndim <= 1 else time.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.chf(time).shape == (m, n)

    def test_cdf(self, regression, time, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.cdf(time).shape == time.shape
        m = 10 if time.ndim <= 1 else time.shape[0]
        n = time.size if time.ndim <= 1 else time.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.cdf(time).shape == (m, n)

    def test_pdf(self, regression, time, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.pdf(time).shape == time.shape
        m = 10 if time.ndim <= 1 else time.shape[0]
        n = time.size if time.ndim <= 1 else time.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.pdf(time).shape == (m, n)

    def test_ppf(self, regression, probability, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.ppf(probability).shape == probability.shape
        m = 1 if probability.ndim <= 1 else probability.shape[0]
        n = probability.size if probability.ndim <= 1 else probability.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.ppf(probability).shape == (m, n)

    def test_ichf(self, regression, probability, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.ichf(probability).shape == probability.shape
        m = 1 if probability.ndim <= 1 else probability.shape[0]
        n = probability.size if probability.ndim <= 1 else probability.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.ichf(probability).shape == (m, n)

    def test_isf(self, regression, probability, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.isf(probability).shape == probability.shape
        assert frozen_regression.isf(np.full(probability.shape, 0.5)) == approx(np.broadcast_to(frozen_regression.median(), probability.shape))
        m = 1 if probability.ndim <= 1 else probability.shape[0]
        n = probability.size if probability.ndim <= 1 else probability.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.isf(probability).shape == (m, n)
        assert frozen_regression.isf(np.full(probability.shape, 0.5)) == approx(np.broadcast_to(frozen_regression.median(), (m, n)))

    def test_dhf(self, regression, time, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.dhf(time).shape == time.shape
        m = 10 if time.ndim <= 1 else time.shape[0]
        n = time.size if time.ndim <= 1 else time.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.dhf(time).shape == (m, n)

    def test_jac_sf(self, regression, time, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.jac_sf(time, asarray=True).shape == (frozen_regression.nb_params,) + time.shape
        m = 10 if time.ndim <= 1 else time.shape[0]
        n = time.size if time.ndim <= 1 else time.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.jac_sf(time, asarray=True).shape == (frozen_regression.nb_params, m, n)

    def test_jac_hf(self, regression, time, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.jac_hf(time, asarray=True).shape == (frozen_regression.nb_params,) + time.shape
        m = 10 if time.ndim <= 1 else time.shape[0]
        n = time.size if time.ndim <= 1 else time.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.jac_hf(time, asarray=True).shape == (frozen_regression.nb_params, m, n)

    def test_jac_chf(self, regression, time, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.jac_chf(time, asarray=True).shape == (frozen_regression.nb_params,) + time.shape
        m = 10 if time.ndim <= 1 else time.shape[0]
        n = time.size if time.ndim <= 1 else time.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.jac_chf(time, asarray=True).shape == (frozen_regression.nb_params, m, n)

    def test_jac_cdf(self, regression, time, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.jac_cdf(time, asarray=True).shape == (frozen_regression.nb_params,) + time.shape
        m = 10 if time.ndim <= 1 else time.shape[0]
        n = time.size if time.ndim <= 1 else time.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.jac_cdf(time, asarray=True).shape == (frozen_regression.nb_params, m, n)

    def test_jac_pdf(self, regression, time, covar):
        frozen_regression = regression.freeze(covar(regression.nb_coef, ))
        assert frozen_regression.jac_pdf(time, asarray=True).shape == (frozen_regression.nb_params,) + time.shape
        m = 10 if time.ndim <= 1 else time.shape[0]
        n = time.size if time.ndim <= 1 else time.shape[1]
        frozen_regression = regression.freeze(covar(m, frozen_regression.nb_coef))
        assert frozen_regression.jac_pdf(time, asarray=True).shape == (frozen_regression.nb_params, m, n)
