# pyright: basic

import numpy as np
from pytest import approx

from relife.lifetime_model import AgeReplacementModel, LeftTruncatedModel


def expected_shape(**kwargs):
    def shape_contrib(**_kwargs):
        yield ()  # yield at least (), in case kwargs is empty
        for k, v in _kwargs.items():
            match k:
                case "covar" if v.ndim == 2:
                    yield v.shape[0], 1
                case "covar" if v.ndim < 2:
                    yield ()
                case "ar" | "a0" if v.ndim == 2 or v.ndim == 0:
                    yield v.shape
                case "ar" | "a0" if v.ndim == 1:
                    yield v.size, 1
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


class TestFrozenRegression:
    def test_rvs(self, regression, covar, rvs_size, rvs_nb_assets):
        frozen_model = regression.freeze(covar)
        assert frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets).shape == rvs_expected_shape(
            rvs_size, nb_assets=rvs_nb_assets, covar=covar
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, covar=covar)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_event=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, covar=covar)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_entry=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, covar=covar)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_event=True, return_entry=True)
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
        assert frozen_model.ppf(probability).shape == expected_shape(probability=probability, covar=covar)

    def test_ichf(self, regression, probability, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.ichf(probability).shape == expected_shape(probability=probability, covar=covar)

    def test_isf(self, regression, probability, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.isf(probability).shape == expected_shape(probability=probability, covar=covar)
        assert frozen_model.isf(np.full(expected_shape(probability=probability, covar=covar), 0.5)) == approx(
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
        assert frozen_model.jac_sf(time, asarray=True).shape == (frozen_model.nb_params,) + expected_shape(
            time=time, covar=covar
        )

    def test_jac_hf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.jac_hf(time, asarray=True).shape == (frozen_model.nb_params,) + expected_shape(
            time=time, covar=covar
        )

    def test_jac_chf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.jac_chf(time, asarray=True).shape == (frozen_model.nb_params,) + expected_shape(
            time=time, covar=covar
        )

    def test_jac_cdf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.jac_cdf(time, asarray=True).shape == (frozen_model.nb_params,) + expected_shape(
            time=time, covar=covar
        )

    def test_jac_pdf(self, regression, time, covar):
        frozen_model = regression.freeze(covar)
        assert frozen_model.jac_pdf(time, asarray=True).shape == (frozen_model.nb_params,) + expected_shape(
            time=time, covar=covar
        )


class TestFrozenAgeReplacementDistribution:
    def test_rvs(self, distribution, ar, rvs_size, rvs_nb_assets):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)

        assert frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets).shape == rvs_expected_shape(
            rvs_size, nb_assets=rvs_nb_assets, ar=ar
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_event=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_entry=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_event=True, return_entry=True)
        )

    def test_sf(self, distribution, time, ar):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        assert frozen_model.sf(time).shape == expected_shape(time=time, ar=ar)

    def test_hf(self, distribution, time, ar):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        assert frozen_model.hf(time).shape == expected_shape(time=time, ar=ar)

    def test_chf(self, distribution, time, ar):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        assert frozen_model.chf(time).shape == expected_shape(time=time, ar=ar)

    def test_cdf(self, distribution, time, ar):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        assert frozen_model.cdf(time).shape == expected_shape(time=time, ar=ar)

    def test_pdf(self, distribution, time, ar):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        assert frozen_model.pdf(time).shape == expected_shape(time=time, ar=ar)

    def test_ppf(self, distribution, time, ar):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        assert frozen_model.ppf(time).shape == expected_shape(time=time, ar=ar)

    def test_ichf(self, distribution, probability, ar):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        assert frozen_model.ichf(probability).shape == expected_shape(probability=probability, ar=ar)

    def test_isf(self, distribution, probability, ar):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        out_shape = expected_shape(probability=probability, ar=ar)
        assert frozen_model.isf(probability).shape == out_shape
        assert frozen_model.isf(np.full(out_shape, 0.5)) == approx(np.broadcast_to(frozen_model.median(), out_shape))

    def test_moment(self, distribution, ar):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        assert frozen_model.moment(1).shape == expected_shape(ar=ar)
        assert frozen_model.moment(2).shape == expected_shape(ar=ar)

    def test_mean(self, distribution, ar):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        assert frozen_model.mean().shape == expected_shape(ar=ar)

    def test_var(self, distribution, ar):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        assert frozen_model.var().shape == expected_shape(ar=ar)

    def test_median(self, distribution, ar):
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        assert frozen_model.median().shape == expected_shape(ar=ar)

    def test_ls_integrate(self, distribution, integration_bound_a, integration_bound_b, ar):
        np.random.seed(10)
        ar = np.random.uniform(2.0, 8.0, size=ar.shape)
        frozen_model = AgeReplacementModel(distribution).freeze(ar)
        # integral_a^b dF(x)
        out_shape = expected_shape(
            integration_bound_a=integration_bound_a,
            integration_bound_b=integration_bound_b,
            ar=ar,
        )
        integration = frozen_model.ls_integrate(np.ones_like, integration_bound_a, integration_bound_b, deg=100)
        assert integration.shape == out_shape
        assert integration == approx(frozen_model.cdf(integration_bound_b) - frozen_model.cdf(integration_bound_a))
        # integral_0^inf x*dF(x)
        integration = frozen_model.ls_integrate(
            lambda x: x,
            np.zeros_like(integration_bound_a),
            np.full_like(integration_bound_b, np.inf),
            deg=100,
        )
        assert integration == approx(np.broadcast_to(frozen_model.mean(), out_shape), rel=1e-3)


class TestFrozenAgeReplacementRegression:
    def test_rvs(self, regression, ar, covar, rvs_size, rvs_nb_assets):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        assert frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets).shape == rvs_expected_shape(
            rvs_size, nb_assets=rvs_nb_assets, ar=ar, covar=covar
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar, covar=covar)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_event=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar, covar=covar)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_entry=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar, covar=covar)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_event=True, return_entry=True)
        )

    def test_sf(self, regression, time, ar, covar):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        assert frozen_model.sf(time).shape == expected_shape(time=time, ar=ar, covar=covar)

    def test_hf(self, regression, time, ar, covar):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        assert frozen_model.hf(time).shape == expected_shape(time=time, ar=ar, covar=covar)

    def test_chf(self, regression, time, ar, covar):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        assert frozen_model.chf(time).shape == expected_shape(time=time, ar=ar, covar=covar)

    def test_cdf(self, regression, time, ar, covar):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        assert frozen_model.cdf(time).shape == expected_shape(time=time, ar=ar, covar=covar)

    def test_pdf(self, regression, time, ar, covar):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        assert frozen_model.pdf(time).shape == expected_shape(time=time, ar=ar, covar=covar)

    def test_ppf(self, regression, time, ar, covar):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        assert frozen_model.ppf(time).shape == expected_shape(time=time, ar=ar, covar=covar)

    def test_ichf(self, regression, probability, ar, covar):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        assert frozen_model.ichf(probability).shape == expected_shape(probability=probability, ar=ar, covar=covar)

    def test_isf(self, regression, probability, ar, covar):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        out_shape = expected_shape(probability=probability, ar=ar, covar=covar)
        assert frozen_model.isf(probability).shape == out_shape
        assert frozen_model.isf(np.full(out_shape, 0.5)) == approx(np.broadcast_to(frozen_model.median(), out_shape))

    def test_moment(self, regression, ar, covar):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        assert frozen_model.moment(1).shape == expected_shape(ar=ar, covar=covar)
        assert frozen_model.moment(2).shape == expected_shape(ar=ar, covar=covar)

    def test_mean(self, regression, ar, covar):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        assert frozen_model.mean().shape == expected_shape(ar=ar, covar=covar)

    def test_var(self, regression, ar, covar):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        assert frozen_model.var().shape == expected_shape(ar=ar, covar=covar)

    def test_median(self, regression, ar, covar):
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        assert frozen_model.median().shape == expected_shape(ar=ar, covar=covar)

    def test_ls_integrate(self, regression, integration_bound_a, integration_bound_b, ar, covar):
        np.random.seed(10)
        ar = np.random.uniform(2.0, 8.0, size=ar.shape)
        frozen_model = AgeReplacementModel(regression).freeze(ar, covar)
        # integral_a^b dF(x)
        out_shape = expected_shape(
            integration_bound_a=integration_bound_a,
            integration_bound_b=integration_bound_b,
            ar=ar,
            covar=covar,
        )
        integration = frozen_model.ls_integrate(np.ones_like, integration_bound_a, integration_bound_b, deg=100)
        assert integration.shape == out_shape
        assert integration == approx(frozen_model.cdf(integration_bound_b) - frozen_model.cdf(integration_bound_a))
        # integral_0^inf x*dF(x)
        integration = frozen_model.ls_integrate(
            lambda x: x,
            np.zeros_like(integration_bound_a),
            np.full_like(integration_bound_b, np.inf),
            deg=100,
        )
        assert integration == approx(np.broadcast_to(frozen_model.mean(), out_shape), rel=1e-3)


class TestFrozenLeftTruncatedDistribution:
    def test_rvs(self, distribution, a0, rvs_size, rvs_nb_assets):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        assert frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets).shape == rvs_expected_shape(
            rvs_size, nb_assets=rvs_nb_assets, a0=a0
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_event=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_entry=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_event=True, return_entry=True)
        )

    def test_sf(self, distribution, time, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        assert frozen_model.sf(time).shape == expected_shape(time=time, a0=a0)

    def test_hf(self, distribution, time, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        assert frozen_model.hf(time).shape == expected_shape(time=time, a0=a0)

    def test_chf(self, distribution, time, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        assert frozen_model.chf(time).shape == expected_shape(time=time, a0=a0)

    def test_cdf(self, distribution, time, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        assert frozen_model.cdf(time).shape == expected_shape(time=time, a0=a0)

    def test_pdf(self, distribution, time, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        assert frozen_model.pdf(time).shape == expected_shape(time=time, a0=a0)

    def test_ppf(self, distribution, time, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        assert frozen_model.ppf(time).shape == expected_shape(time=time, a0=a0)

    def test_ichf(self, distribution, probability, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        assert frozen_model.ichf(probability).shape == expected_shape(probability=probability, a0=a0)

    def test_isf(self, distribution, probability, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        out_shape = expected_shape(probability=probability, a0=a0)
        assert frozen_model.isf(probability).shape == out_shape
        assert frozen_model.isf(np.full(out_shape, 0.5)) == approx(np.broadcast_to(frozen_model.median(), out_shape))

    def test_moment(self, distribution, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        assert frozen_model.moment(1).shape == expected_shape(a0=a0)
        assert frozen_model.moment(2).shape == expected_shape(a0=a0)

    def test_mean(self, distribution, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        assert frozen_model.mean().shape == expected_shape(a0=a0)

    def test_var(self, distribution, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        assert frozen_model.var().shape == expected_shape(a0=a0)

    def test_median(self, distribution, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        assert frozen_model.median().shape == expected_shape(a0=a0)

    def test_ls_integrate(self, distribution, integration_bound_a, integration_bound_b, a0):
        frozen_model = LeftTruncatedModel(distribution).freeze(a0)
        # integral_a^b dF(x)
        out_shape = expected_shape(
            integration_bound_a=integration_bound_a,
            integration_bound_b=integration_bound_b,
            a0=a0,
        )
        integration = frozen_model.ls_integrate(np.ones_like, integration_bound_a, integration_bound_b, deg=100)
        assert integration.shape == out_shape
        assert integration == approx(frozen_model.cdf(integration_bound_b) - frozen_model.cdf(integration_bound_a))
        # integral_0^inf x*dF(x)
        integration = frozen_model.ls_integrate(
            lambda x: x,
            np.zeros_like(integration_bound_a),
            np.full_like(integration_bound_b, np.inf),
            deg=100,
        )
        assert integration == approx(np.broadcast_to(frozen_model.mean(), out_shape), rel=1e-3)


class TestLeftTruncatedRegression:
    # def test_args_nb_assets(self, regression):
    #     frozen_distribution = regression.freeze()
    #     assert frozen_distribution.args_nb_assets == 1

    def test_rvs(self, regression, a0, covar, rvs_size, rvs_nb_assets):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        assert frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets).shape == rvs_expected_shape(
            rvs_size, nb_assets=rvs_nb_assets, a0=a0, covar=covar
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0, covar=covar)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_event=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0, covar=covar)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_entry=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0, covar=covar)
            for arr in frozen_model.rvs(rvs_size, nb_assets=rvs_nb_assets, return_event=True, return_entry=True)
        )

    def test_sf(self, regression, time, a0, covar):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        assert frozen_model.sf(time).shape == expected_shape(time=time, a0=a0, covar=covar)

    def test_hf(self, regression, time, a0, covar):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        assert frozen_model.hf(time).shape == expected_shape(time=time, a0=a0, covar=covar)

    def test_chf(self, regression, time, a0, covar):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        assert frozen_model.chf(time).shape == expected_shape(time=time, a0=a0, covar=covar)

    def test_cdf(self, regression, time, a0, covar):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        assert frozen_model.cdf(time).shape == expected_shape(time=time, a0=a0, covar=covar)

    def test_pdf(self, regression, time, a0, covar):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        assert frozen_model.pdf(time).shape == expected_shape(time=time, a0=a0, covar=covar)

    def test_ppf(self, regression, time, a0, covar):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        assert frozen_model.ppf(time).shape == expected_shape(time=time, a0=a0, covar=covar)

    def test_ichf(self, regression, probability, a0, covar):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        assert frozen_model.ichf(probability).shape == expected_shape(probability=probability, a0=a0, covar=covar)

    def test_isf(self, regression, probability, a0, covar):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        out_shape = expected_shape(probability=probability, a0=a0, covar=covar)
        assert frozen_model.isf(probability).shape == out_shape
        assert frozen_model.isf(np.full(out_shape, 0.5)) == approx(np.broadcast_to(frozen_model.median(), out_shape))

    def test_moment(self, regression, a0, covar):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        assert frozen_model.moment(1).shape == expected_shape(a0=a0, covar=covar)
        assert frozen_model.moment(2).shape == expected_shape(a0=a0, covar=covar)

    def test_mean(self, regression, a0, covar):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        assert frozen_model.mean().shape == expected_shape(a0=a0, covar=covar)

    def test_var(self, regression, a0, covar):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        assert frozen_model.var().shape == expected_shape(a0=a0, covar=covar)

    def test_median(self, regression, a0, covar):
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        assert frozen_model.median().shape == expected_shape(a0=a0, covar=covar)

    def test_ls_integrate(self, regression, integration_bound_a, integration_bound_b, a0, covar):
        np.random.seed(10)
        a0 = np.random.uniform(2.0, 8.0, size=a0.shape)
        frozen_model = LeftTruncatedModel(regression).freeze(a0, covar)
        # integral_a^b dF(x)
        out_shape = expected_shape(
            integration_bound_a=integration_bound_a,
            integration_bound_b=integration_bound_b,
            a0=a0,
            covar=covar,
        )
        integration = frozen_model.ls_integrate(np.ones_like, integration_bound_a, integration_bound_b, deg=100)
        assert integration.shape == out_shape
        assert integration == approx(frozen_model.cdf(integration_bound_b) - frozen_model.cdf(integration_bound_a))
        # integral_0^inf x*dF(x)
        integration = frozen_model.ls_integrate(
            lambda x: x,
            np.zeros_like(integration_bound_a),
            np.full_like(integration_bound_b, np.inf),
            deg=100,
        )
        assert integration == approx(np.broadcast_to(frozen_model.mean(), out_shape), rel=1e-3)
