import random

import numpy as np
import pytest

from relife.lifetime_model import Weibull, Gompertz, ProportionalHazard, AFT, Gamma, LogLogistic


M = 2
N = 10


@pytest.mark.parametrize(
    "model", [
        Weibull(2, 0.05),
        Gompertz(0.01, 0.1),
        Gamma(2, 0.05),
        LogLogistic(3, 0.05),
    ]
)
class Test0DDistribution:
    time = random.uniform(2.5, 10.0)
    probability = random.random()
    cumulative_hazard_rate = random.uniform(2.5, 10.0)

    def test_sf(self, model):
        assert model.sf(self.time).shape == ()

    def test_hf(self, model):
        assert model.hf(self.time).shape == ()

    def test_chf(self, model):
        assert model.chf(self.time).shape == ()

    def test_cdf(self, model):
        assert model.cdf(self.time).shape == ()

    def test_pdf(self, model):
        assert model.pdf(self.time).shape == ()

    def test_ppf(self, model):
        assert model.ppf(self.probability).shape == ()

    def test_ichf(self, model):
        assert model.ichf(self.probability).shape == ()

    # def test_mrl(self, model):
    #     assert model.mrl(self.time).shape == ()

    # def test_mean(self, model):
    #     assert model.mean().shape == ()
    #
    # def test_var(self, model):
    #     assert model.var().shape == ()
    #
    # def test_median(self, model):
    #     assert model.median().shape == ()




@pytest.mark.parametrize(
    "model", [
        Weibull(2, 0.05),
        Gompertz(0.01, 0.1),
        Gamma(2, 0.05),
        LogLogistic(3, 0.05),
    ]
)
class Test1DDistribution:
    time = np.random.uniform(2.5, 10.0, size=(M,))
    probability = np.random.uniform(0, 1, size=(M,))
    cumulative_hazard_rate = np.random.uniform(2.5, 10.0, size=(M,))

    def test_sf(self, model):
        assert model.sf(self.time).shape == (M,)

    def test_hf(self, model):
        assert model.hf(self.time).shape == (M,)

    def test_chf(self, model):
        assert model.chf(self.time).shape == (M,)

    def test_cdf(self, model):
        assert model.cdf(self.time).shape == (M,)

    def test_pdf(self, model):
        assert model.pdf(self.time).shape == (M,)

    def test_ppf(self, model):
        assert model.ppf(self.probability).shape == (M,)

    def test_ichf(self, model):
        assert model.ichf(self.probability).shape == (M,)

    # def test_mrl(self, model):
    #     assert model.mrl(self.time).shape == (M,)
    #
    # def test_mean(self, model):
    #     assert model.mean().shape == (M,)
    #
    # def test_var(self, model):
    #     assert model.var().shape == (M,)
    #
    # def test_median(self, model):
    #     assert model.median().shape == (M,)




@pytest.mark.parametrize(
    "model", [
        Weibull(2, 0.05),
        Gompertz(0.01, 0.1),
        Gamma(2, 0.05),
        LogLogistic(3, 0.05),
    ]
)
class Test2DDistribution:
    time = np.random.uniform(2.5, 10.0, size=(M,N))
    probability = np.random.uniform(0, 1, size=(M,N))
    cumulative_hazard_rate = np.random.uniform(2.5, 10.0, size=(M,N))

    def test_sf(self, model):
        assert model.sf(self.time).shape == (M,N)

    def test_hf(self, model):
        assert model.hf(self.time).shape == (M,N)

    def test_chf(self, model):
        assert model.chf(self.time).shape == (M,N)

    def test_cdf(self, model):
        assert model.cdf(self.time).shape == (M,N)

    def test_pdf(self, model):
        assert model.pdf(self.time).shape == (M,N)

    def test_ppf(self, model):
        assert model.ppf(self.probability).shape == (M,N)

    def test_ichf(self, model):
        assert model.ichf(self.probability).shape == (M,N)

    # def test_mrl(self, model):
    #     assert model.mrl(self.time).shape == (M,N)
    #
    # def test_mean(self, model):
    #     assert model.mean().shape == (M,N)
    #
    # def test_var(self, model):
    #     assert model.var().shape == (M,N)
    #
    # def test_median(self, model):
    #     assert model.median().shape == (M,N)
    #



NB_COEF = 4
@pytest.mark.parametrize(
    "model",
    [
        ProportionalHazard(
            Weibull(7, 0.05), coef=(random.uniform(1.0, 2.0),) * NB_COEF
        ),
        AFT(Weibull(7, 0.05), coef=(random.uniform(1.0, 2.0),) * NB_COEF),
        ProportionalHazard(
            Weibull(7, 0.05), coef=(random.uniform(1.0, 2.0),) * NB_COEF
        ),
        AFT(Weibull(7, 0.05), coef=(random.uniform(1.0, 2.0),) * NB_COEF),
    ],
)
class Test0DRegression:
    time = random.uniform(2.5, 10.0)
    covar = np.random.randn(M, NB_COEF)

    def test_sf(self, model):
        assert model.sf(self.time, self.covar).shape == (M,)

    def test_hf(self, model):
        assert model.hf(self.time, self.covar).shape == (M,)

    def test_chf(self, model):
        assert model.chf(self.time, self.covar).shape == (M,)

    def test_cdf(self, model):
        assert model.cdf(self.time, self.covar).shape == (M,)

    def test_pdf(self, model):
        assert model.pdf(self.time, self.covar).shape == (M,)



@pytest.mark.parametrize(
    "model",
    [
        ProportionalHazard(
            Weibull(7, 0.05), coef=(random.uniform(1.0, 2.0),) * NB_COEF
        ),
        AFT(Weibull(7, 0.05), coef=(random.uniform(1.0, 2.0),) * NB_COEF),
    ],
)
class Test1DRegression:
    time = np.random.uniform(2.5, 10.0, size=(M,))
    covar = np.random.randn(M, NB_COEF)

    def test_sf(self, model):
        assert model.sf(self.time, self.covar).shape == (M,)

    def test_hf(self, model):
        assert model.hf(self.time, self.covar).shape == (M,)

    def test_chf(self, model):
        assert model.chf(self.time, self.covar).shape == (M,)

    def test_cdf(self, model):
        assert model.cdf(self.time, self.covar).shape == (M,)

    def test_pdf(self, model):
        assert model.pdf(self.time, self.covar).shape == (M,)



@pytest.mark.parametrize(
    "model",
    [
        ProportionalHazard(
            Weibull(7, 0.05), coef=(random.uniform(1.0, 2.0),) * NB_COEF
        ),
        AFT(Weibull(7, 0.05), coef=(random.uniform(1.0, 2.0),) * NB_COEF),
    ],
)
class Test2DRegression:
    time = np.random.uniform(2.5, 10.0, size=(M, N))
    covar = np.random.randn(M, NB_COEF)

    def test_sf(self, model):
        assert model.sf(self.time, self.covar).shape == (M,N)

    def test_hf(self, model):
        assert model.hf(self.time, self.covar).shape == (M,N)

    def test_chf(self, model):
        assert model.chf(self.time, self.covar).shape == (M,N)

    def test_cdf(self, model):
        assert model.cdf(self.time, self.covar).shape == (M,N)

    def test_pdf(self, model):
        assert model.pdf(self.time, self.covar).shape == (M,N)

