import pytest


class TestDistribution:

    # def test_rvs(distribution, size):
    #     m, n = 3, 10
    #     assert distribution.rvs(seed=21).shape == ()
    #     assert distribution.rvs(n, seed=21).shape == (n,)
    #     assert distribution.rvs((n,), seed=21).shape == (n,)
    #     assert distribution.rvs((m, 1), seed=21).shape == (m, 1)
    #     assert distribution.rvs((m, n), seed=21).shape == (m, n)

    def test_sample_lifetime_data(self, distribution, size):
        pass


class TestRegression:
    ...
    # def test_rvs(regression, covar):
    #     match covar.shape:
    #         case (_,):
    #             m, n = 10, 20
    #             assert regression.rvs(covar, seed=21).shape == ()
    #             assert regression.rvs(covar, size=n, seed=21).shape == (n,)
    #             assert regression.rvs(covar, size=(n,), seed=21).shape == (n,)
    #             assert regression.rvs(covar, size=(m, n), seed=21).shape == (m, n)
    #         case (1, _):
    #             m, n = 10, 20
    #             assert regression.rvs(covar, seed=21).shape == (1, 1)
    #             assert regression.rvs(covar, size=n, seed=21).shape == (1, n)
    #             assert regression.rvs(covar, size=(n,), seed=21).shape == (1, n)
    #             assert regression.rvs(covar, size=(m, n), seed=21).shape == (m, n)
    #         case (m, _):
    #             n = 20
    #             assert regression.rvs(covar, seed=21).shape == (m, 1)
    #             assert regression.rvs(covar, size=n, seed=21).shape == (m, n)
    #             assert regression.rvs(covar, size=(n,), seed=21).shape == (m, n)
    #             assert regression.rvs(covar, size=(m, n), seed=21).shape == (m, n)
