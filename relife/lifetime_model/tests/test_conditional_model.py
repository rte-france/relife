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


class TestAgeReplacementDistribution:
    def test_rvs(self, distribution, ar, rvs_size, rvs_nb_assets):
        assert AgeReplacementModel(distribution).rvs(rvs_size, ar, nb_assets=rvs_nb_assets).shape == rvs_expected_shape(
            rvs_size, nb_assets=rvs_nb_assets, ar=ar
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar)
            for arr in AgeReplacementModel(distribution).rvs(rvs_size, ar, nb_assets=rvs_nb_assets, return_event=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar)
            for arr in AgeReplacementModel(distribution).rvs(rvs_size, ar, nb_assets=rvs_nb_assets, return_entry=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar)
            for arr in AgeReplacementModel(distribution).rvs(
                rvs_size,
                ar,
                nb_assets=rvs_nb_assets,
                return_event=True,
                return_entry=True,
            )
        )

    def test_sf(self, distribution, time, ar):
        assert AgeReplacementModel(distribution).sf(time, ar).shape == expected_shape(time=time, ar=ar)

    def test_hf(self, distribution, time, ar):
        assert AgeReplacementModel(distribution).hf(time, ar).shape == expected_shape(time=time, ar=ar)

    def test_chf(self, distribution, time, ar):
        assert AgeReplacementModel(distribution).chf(time, ar).shape == expected_shape(time=time, ar=ar)

    def test_cdf(self, distribution, time, ar):
        assert AgeReplacementModel(distribution).cdf(time, ar).shape == expected_shape(time=time, ar=ar)

    def test_pdf(self, distribution, time, ar):
        assert AgeReplacementModel(distribution).pdf(time, ar).shape == expected_shape(time=time, ar=ar)

    def test_ppf(self, distribution, time, ar):
        assert AgeReplacementModel(distribution).ppf(time, ar).shape == expected_shape(time=time, ar=ar)

    def test_ichf(self, distribution, probability, ar):
        assert AgeReplacementModel(distribution).ichf(probability, ar).shape == expected_shape(
            probability=probability, ar=ar
        )

    def test_isf(self, distribution, probability, ar):
        out_shape = expected_shape(probability=probability, ar=ar)
        assert AgeReplacementModel(distribution).isf(probability, ar).shape == out_shape
        assert AgeReplacementModel(distribution).isf(np.full(out_shape, 0.5), ar) == approx(
            np.broadcast_to(AgeReplacementModel(distribution).median(ar), out_shape)
        )

    def test_moment(self, distribution, ar):
        assert AgeReplacementModel(distribution).moment(1, ar).shape == expected_shape(ar=ar)
        assert AgeReplacementModel(distribution).moment(2, ar).shape == expected_shape(ar=ar)

    def test_mean(self, distribution, ar):
        assert AgeReplacementModel(distribution).mean(ar).shape == expected_shape(ar=ar)

    def test_var(self, distribution, ar):
        assert AgeReplacementModel(distribution).var(ar).shape == expected_shape(ar=ar)

    def test_median(self, distribution, ar):
        assert AgeReplacementModel(distribution).median(ar).shape == expected_shape(ar=ar)

    def test_ls_integrate(self, distribution, integration_bound_a, integration_bound_b, ar):
        np.random.seed(10)
        ar = np.random.uniform(2.0, 8.0, size=ar.shape)
        # integral_a^b dF(x)
        out_shape = expected_shape(
            integration_bound_a=integration_bound_a,
            integration_bound_b=integration_bound_b,
            ar=ar,
        )
        integration = AgeReplacementModel(distribution).ls_integrate(
            np.ones_like, integration_bound_a, integration_bound_b, ar, deg=100
        )
        assert integration.shape == out_shape
        assert integration == approx(
            AgeReplacementModel(distribution).cdf(integration_bound_b, ar)
            - AgeReplacementModel(distribution).cdf(integration_bound_a, ar)
        )
        # integral_0^inf x*dF(x)
        integration = AgeReplacementModel(distribution).ls_integrate(
            lambda x: x,
            np.zeros_like(integration_bound_a),
            np.full_like(integration_bound_b, np.inf),
            ar,
            deg=100,
        )
        assert integration == approx(
            np.broadcast_to(AgeReplacementModel(distribution).mean(ar), out_shape),
            rel=1e-3,
        )


class TestAgeReplacementRegression:
    def test_rvs(self, regression, ar, covar, rvs_size, rvs_nb_assets):
        assert AgeReplacementModel(regression).rvs(
            rvs_size, ar, covar, nb_assets=rvs_nb_assets
        ).shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar, covar=covar)
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar, covar=covar)
            for arr in AgeReplacementModel(regression).rvs(
                rvs_size, ar, covar, nb_assets=rvs_nb_assets, return_event=True
            )
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar, covar=covar)
            for arr in AgeReplacementModel(regression).rvs(
                rvs_size, ar, covar, nb_assets=rvs_nb_assets, return_entry=True
            )
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, ar=ar, covar=covar)
            for arr in AgeReplacementModel(regression).rvs(
                rvs_size,
                ar,
                covar,
                nb_assets=rvs_nb_assets,
                return_event=True,
                return_entry=True,
            )
        )

    def test_sf(self, regression, time, ar, covar):
        assert AgeReplacementModel(regression).sf(time, ar, covar).shape == expected_shape(
            time=time, ar=ar, covar=covar
        )

    def test_hf(self, regression, time, ar, covar):
        assert AgeReplacementModel(regression).hf(time, ar, covar).shape == expected_shape(
            time=time, ar=ar, covar=covar
        )

    def test_chf(self, regression, time, ar, covar):
        assert AgeReplacementModel(regression).chf(time, ar, covar).shape == expected_shape(
            time=time, ar=ar, covar=covar
        )

    def test_cdf(self, regression, time, ar, covar):
        assert AgeReplacementModel(regression).cdf(time, ar, covar).shape == expected_shape(
            time=time, ar=ar, covar=covar
        )

    def test_pdf(self, regression, time, ar, covar):
        assert AgeReplacementModel(regression).pdf(time, ar, covar).shape == expected_shape(
            time=time, ar=ar, covar=covar
        )

    def test_ppf(self, regression, time, ar, covar):
        assert AgeReplacementModel(regression).ppf(time, ar, covar).shape == expected_shape(
            time=time, ar=ar, covar=covar
        )

    def test_ichf(self, regression, probability, ar, covar):
        assert AgeReplacementModel(regression).ichf(probability, ar, covar).shape == expected_shape(
            probability=probability, ar=ar, covar=covar
        )

    def test_isf(self, regression, probability, ar, covar):
        out_shape = expected_shape(probability=probability, ar=ar, covar=covar)
        assert AgeReplacementModel(regression).isf(probability, ar, covar).shape == out_shape
        assert AgeReplacementModel(regression).isf(np.full(out_shape, 0.5), ar, covar) == approx(
            np.broadcast_to(AgeReplacementModel(regression).median(ar, covar), out_shape)
        )

    def test_moment(self, regression, ar, covar):
        assert AgeReplacementModel(regression).moment(1, ar, covar).shape == expected_shape(ar=ar, covar=covar)
        assert AgeReplacementModel(regression).moment(2, ar, covar).shape == expected_shape(ar=ar, covar=covar)

    def test_mean(self, regression, ar, covar):
        assert AgeReplacementModel(regression).mean(ar, covar).shape == expected_shape(ar=ar, covar=covar)

    def test_var(self, regression, ar, covar):
        assert AgeReplacementModel(regression).var(ar, covar).shape == expected_shape(ar=ar, covar=covar)

    def test_median(self, regression, ar, covar):
        assert AgeReplacementModel(regression).median(ar, covar).shape == expected_shape(ar=ar, covar=covar)

    def test_ls_integrate(self, regression, integration_bound_a, integration_bound_b, ar, covar):
        np.random.seed(10)
        ar = np.random.uniform(2.0, 8.0, size=ar.shape)
        # integral_a^b dF(x)
        out_shape = expected_shape(
            integration_bound_a=integration_bound_a,
            integration_bound_b=integration_bound_b,
            ar=ar,
            covar=covar,
        )
        integration = AgeReplacementModel(regression).ls_integrate(
            np.ones_like, integration_bound_a, integration_bound_b, ar, covar, deg=100
        )
        assert integration.shape == out_shape
        assert integration == approx(
            AgeReplacementModel(regression).cdf(integration_bound_b, ar, covar)
            - AgeReplacementModel(regression).cdf(integration_bound_a, ar, covar)
        )
        # integral_0^inf x*dF(x)
        integration = AgeReplacementModel(regression).ls_integrate(
            lambda x: x,
            np.zeros_like(integration_bound_a),
            np.full_like(integration_bound_b, np.inf),
            ar,
            covar,
            deg=100,
        )
        assert integration == approx(
            np.broadcast_to(AgeReplacementModel(regression).mean(ar, covar), out_shape),
            rel=1e-3,
        )


class TestLeftTruncatedDistribution:
    def test_rvs(self, distribution, a0, rvs_size, rvs_nb_assets):
        assert LeftTruncatedModel(distribution).rvs(rvs_size, a0, nb_assets=rvs_nb_assets).shape == rvs_expected_shape(
            rvs_size, nb_assets=rvs_nb_assets, a0=a0
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0)
            for arr in LeftTruncatedModel(distribution).rvs(rvs_size, a0, nb_assets=rvs_nb_assets, return_event=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0)
            for arr in LeftTruncatedModel(distribution).rvs(rvs_size, a0, nb_assets=rvs_nb_assets, return_entry=True)
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0)
            for arr in LeftTruncatedModel(distribution).rvs(
                rvs_size,
                a0,
                nb_assets=rvs_nb_assets,
                return_event=True,
                return_entry=True,
            )
        )

    def test_sf(self, distribution, time, a0):
        assert LeftTruncatedModel(distribution).sf(time, a0).shape == expected_shape(time=time, a0=a0)

    def test_hf(self, distribution, time, a0):
        assert LeftTruncatedModel(distribution).hf(time, a0).shape == expected_shape(time=time, a0=a0)

    def test_chf(self, distribution, time, a0):
        assert LeftTruncatedModel(distribution).chf(time, a0).shape == expected_shape(time=time, a0=a0)

    def test_cdf(self, distribution, time, a0):
        assert LeftTruncatedModel(distribution).cdf(time, a0).shape == expected_shape(time=time, a0=a0)

    def test_pdf(self, distribution, time, a0):
        assert LeftTruncatedModel(distribution).pdf(time, a0).shape == expected_shape(time=time, a0=a0)

    def test_ppf(self, distribution, time, a0):
        assert LeftTruncatedModel(distribution).ppf(time, a0).shape == expected_shape(time=time, a0=a0)

    def test_ichf(self, distribution, probability, a0):
        assert LeftTruncatedModel(distribution).ichf(probability, a0).shape == expected_shape(
            probability=probability, a0=a0
        )

    def test_isf(self, distribution, probability, a0):
        out_shape = expected_shape(probability=probability, a0=a0)
        assert LeftTruncatedModel(distribution).isf(probability, a0).shape == out_shape
        assert LeftTruncatedModel(distribution).isf(np.full(out_shape, 0.5), a0) == approx(
            np.broadcast_to(LeftTruncatedModel(distribution).median(a0), out_shape)
        )

    def test_moment(self, distribution, a0):
        assert LeftTruncatedModel(distribution).moment(1, a0).shape == expected_shape(a0=a0)
        assert LeftTruncatedModel(distribution).moment(2, a0).shape == expected_shape(a0=a0)

    def test_mean(self, distribution, a0):
        assert LeftTruncatedModel(distribution).mean(a0).shape == expected_shape(a0=a0)

    def test_var(self, distribution, a0):
        assert LeftTruncatedModel(distribution).var(a0).shape == expected_shape(a0=a0)

    def test_median(self, distribution, a0):
        assert LeftTruncatedModel(distribution).median(a0).shape == expected_shape(a0=a0)

    def test_ls_integrate(self, distribution, integration_bound_a, integration_bound_b, a0):
        # integral_a^b dF(x)
        out_shape = expected_shape(
            integration_bound_a=integration_bound_a,
            integration_bound_b=integration_bound_b,
            a0=a0,
        )
        integration = LeftTruncatedModel(distribution).ls_integrate(
            np.ones_like, integration_bound_a, integration_bound_b, a0, deg=100
        )
        assert integration.shape == out_shape
        assert integration == approx(
            LeftTruncatedModel(distribution).cdf(integration_bound_b, a0)
            - LeftTruncatedModel(distribution).cdf(integration_bound_a, a0)
        )
        # integral_0^inf x*dF(x)
        integration = LeftTruncatedModel(distribution).ls_integrate(
            lambda x: x,
            np.zeros_like(integration_bound_a),
            np.full_like(integration_bound_b, np.inf),
            a0,
            deg=100,
        )
        assert integration == approx(
            np.broadcast_to(LeftTruncatedModel(distribution).mean(a0), out_shape),
            rel=1e-3,
        )


class TestLeftTruncatedRegression:
    def test_rvs(self, regression, a0, covar, rvs_size, rvs_nb_assets):
        assert LeftTruncatedModel(regression).rvs(
            rvs_size, a0, covar, nb_assets=rvs_nb_assets
        ).shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0, covar=covar)
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0, covar=covar)
            for arr in LeftTruncatedModel(regression).rvs(
                rvs_size, a0, covar, nb_assets=rvs_nb_assets, return_event=True
            )
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0, covar=covar)
            for arr in LeftTruncatedModel(regression).rvs(
                rvs_size, a0, covar, nb_assets=rvs_nb_assets, return_entry=True
            )
        )
        assert all(
            arr.shape == rvs_expected_shape(rvs_size, nb_assets=rvs_nb_assets, a0=a0, covar=covar)
            for arr in LeftTruncatedModel(regression).rvs(
                rvs_size,
                a0,
                covar,
                nb_assets=rvs_nb_assets,
                return_event=True,
                return_entry=True,
            )
        )

    def test_sf(self, regression, time, a0, covar):
        assert LeftTruncatedModel(regression).sf(time, a0, covar).shape == expected_shape(time=time, a0=a0, covar=covar)

    def test_hf(self, regression, time, a0, covar):
        assert LeftTruncatedModel(regression).hf(time, a0, covar).shape == expected_shape(time=time, a0=a0, covar=covar)

    def test_chf(self, regression, time, a0, covar):
        assert LeftTruncatedModel(regression).chf(time, a0, covar).shape == expected_shape(
            time=time, a0=a0, covar=covar
        )

    def test_cdf(self, regression, time, a0, covar):
        assert LeftTruncatedModel(regression).cdf(time, a0, covar).shape == expected_shape(
            time=time, a0=a0, covar=covar
        )

    def test_pdf(self, regression, time, a0, covar):
        assert LeftTruncatedModel(regression).pdf(time, a0, covar).shape == expected_shape(
            time=time, a0=a0, covar=covar
        )

    def test_ppf(self, regression, time, a0, covar):
        assert LeftTruncatedModel(regression).ppf(time, a0, covar).shape == expected_shape(
            time=time, a0=a0, covar=covar
        )

    def test_ichf(self, regression, probability, a0, covar):
        assert LeftTruncatedModel(regression).ichf(probability, a0, covar).shape == expected_shape(
            probability=probability, a0=a0, covar=covar
        )

    def test_isf(self, regression, probability, a0, covar):
        out_shape = expected_shape(probability=probability, a0=a0, covar=covar)
        assert LeftTruncatedModel(regression).isf(probability, a0, covar).shape == out_shape
        assert LeftTruncatedModel(regression).isf(np.full(out_shape, 0.5), a0, covar) == approx(
            np.broadcast_to(LeftTruncatedModel(regression).median(a0, covar), out_shape)
        )

    def test_moment(self, regression, a0, covar):
        assert LeftTruncatedModel(regression).moment(1, a0, covar).shape == expected_shape(a0=a0, covar=covar)
        assert LeftTruncatedModel(regression).moment(2, a0, covar).shape == expected_shape(a0=a0, covar=covar)

    def test_mean(self, regression, a0, covar):
        assert LeftTruncatedModel(regression).mean(a0, covar).shape == expected_shape(a0=a0, covar=covar)

    def test_var(self, regression, a0, covar):
        assert LeftTruncatedModel(regression).var(a0, covar).shape == expected_shape(a0=a0, covar=covar)

    def test_median(self, regression, a0, covar):
        assert LeftTruncatedModel(regression).median(a0, covar).shape == expected_shape(a0=a0, covar=covar)

    def test_ls_integrate(self, regression, integration_bound_a, integration_bound_b, a0, covar):
        # integral_a^b dF(x)
        out_shape = expected_shape(
            integration_bound_a=integration_bound_a,
            integration_bound_b=integration_bound_b,
            a0=a0,
            covar=covar,
        )
        np.random.seed(10)
        a0 = np.random.uniform(2.0, 8.0, size=a0.shape)
        integration = LeftTruncatedModel(regression).ls_integrate(
            np.ones_like, integration_bound_a, integration_bound_b, a0, covar, deg=100
        )
        assert integration.shape == out_shape
        assert integration == approx(
            LeftTruncatedModel(regression).cdf(integration_bound_b, a0, covar)
            - LeftTruncatedModel(regression).cdf(integration_bound_a, a0, covar)
        )
        # integral_0^inf x*dF(x)
        integration = LeftTruncatedModel(regression).ls_integrate(
            lambda x: x,
            np.zeros_like(integration_bound_a),
            np.full_like(integration_bound_b, np.inf),
            a0,
            covar,
            deg=100,
        )
        assert integration == approx(
            np.broadcast_to(LeftTruncatedModel(regression).mean(a0, covar), out_shape),
            rel=1e-3,
        )
