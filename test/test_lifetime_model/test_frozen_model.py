from pytest import approx
import numpy as np


class TestFrozenDistribution:
    # def test_args_nb_assets(self, distribution):
    #     frozen_distribution = distribution.freeze()
    #     assert frozen_distribution.args_nb_assets == 1

    def test_sf(self, distribution, time, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.sf(time).shape == time.shape
        assert frozen_distribution.sf(np.full(time.shape, distribution.median())) == approx(
            np.full(time.shape, 0.5), rel=1e-3
        )

    def test_hf(self, distribution, time, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.hf(time).shape == expected_out_shape(time=time)

    def test_chf(self, distribution, time, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.chf(time).shape == expected_out_shape(time=time)

    def test_cdf(self, distribution, time, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.cdf(time).shape == expected_out_shape(time=time)

    def test_pdf(self, distribution, time, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.pdf(time).shape == expected_out_shape(time=time)

    def test_ppf(self, distribution, probability, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.ppf(probability).shape == expected_out_shape(probability=probability)

    def test_ichf(self, distribution, probability, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.ichf(probability).shape == expected_out_shape(probability=probability)

    def test_isf(self, distribution, probability, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.isf(probability).shape == expected_out_shape(probability=probability)
        assert frozen_distribution.isf(np.full(expected_out_shape(probability=probability), 0.5)) == approx(
            np.full(expected_out_shape(probability=probability), distribution.median())
        )

    def test_moment(self, distribution, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.moment(1).shape == expected_out_shape()
        assert frozen_distribution.moment(2).shape == expected_out_shape()

    def test_mean(self, distribution, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.mean().shape == expected_out_shape()

    def test_var(self, distribution, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.var().shape == expected_out_shape()

    def test_median(self, distribution, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.median().shape == expected_out_shape()

    def test_dhf(self, distribution, time, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.dhf(time).shape == expected_out_shape(time=time)

    def test_jac_sf(self, distribution, time, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.jac_sf(time, asarray=True).shape == (
            frozen_distribution.nb_params,
        ) + expected_out_shape(time=time)

    def test_jac_hf(self, distribution, time, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.jac_hf(time, asarray=True).shape == (
            frozen_distribution.nb_params,
        ) + expected_out_shape(time=time)

    def test_jac_chf(self, distribution, time, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.jac_chf(time, asarray=True).shape == (
            frozen_distribution.nb_params,
        ) + expected_out_shape(time=time)

    def test_jac_cdf(self, distribution, time, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.jac_cdf(time, asarray=True).shape == (
            frozen_distribution.nb_params,
        ) + expected_out_shape(time=time)

    def test_jac_pdf(self, distribution, time, expected_out_shape):
        frozen_distribution = distribution.freeze()
        assert frozen_distribution.jac_pdf(time, asarray=True).shape == (
            frozen_distribution.nb_params,
        ) + expected_out_shape(time=time)


class TestFrozenRegression:
    # def test_args_nb_assets(self, regression, covar):
    #     frozen_regression = regression.freeze(covar)
    #     assert frozen_regression.args_nb_assets == 1
    #     frozen_regression = regression.freeze(covar(10, regression.nb_coef))
    #     assert frozen_regression.args_nb_assets == 10

    # def test_rvs(self, regression, covar):
    #     m, n = 10, 3
    #     frozen_regression = regression.freeze(covar)
    #     assert frozen_regression.rvs(seed=21).shape == ()
    #     assert frozen_regression.rvs(size=(m, 1), seed=21).shape == (m, 1)
    #     assert frozen_regression.rvs(size=(m, n), seed=21).shape == (m, n)

    def test_sf(self, regression, time, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.sf(time).shape == expected_out_shape(time=time, covar=covar)

    def test_hf(self, regression, time, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.hf(time).shape == expected_out_shape(time=time, covar=covar)

    def test_chf(self, regression, time, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.chf(time).shape == expected_out_shape(time=time, covar=covar)

    def test_cdf(self, regression, time, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.cdf(time).shape == expected_out_shape(time=time, covar=covar)

    def test_pdf(self, regression, time, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.pdf(time).shape == expected_out_shape(time=time, covar=covar)

    def test_ppf(self, regression, probability, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.ppf(probability).shape == expected_out_shape(probability=probability, covar=covar)

    def test_ichf(self, regression, probability, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.ichf(probability).shape == expected_out_shape(probability=probability, covar=covar)

    def test_isf(self, regression, probability, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.isf(probability).shape == expected_out_shape(probability=probability, covar=covar)
        assert frozen_regression.isf(np.full(expected_out_shape(probability=probability, covar=covar), 0.5)) == approx(
            np.broadcast_to(frozen_regression.median(), expected_out_shape(probability=probability, covar=covar))
        )

    def test_dhf(self, regression, time, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.dhf(time).shape == expected_out_shape(time=time, covar=covar)

    def test_jac_sf(self, regression, time, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.jac_sf(time, asarray=True).shape == (
            frozen_regression.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_jac_hf(self, regression, time, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.jac_hf(time, asarray=True).shape == (
            frozen_regression.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_jac_chf(self, regression, time, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.jac_chf(time, asarray=True).shape == (
            frozen_regression.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_jac_cdf(self, regression, time, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.jac_cdf(time, asarray=True).shape == (
            frozen_regression.nb_params,
        ) + expected_out_shape(time=time, covar=covar)

    def test_jac_pdf(self, regression, time, covar, expected_out_shape):
        frozen_regression = regression.freeze(covar)
        assert frozen_regression.jac_pdf(time, asarray=True).shape == (
            frozen_regression.nb_params,
        ) + expected_out_shape(time=time, covar=covar)
