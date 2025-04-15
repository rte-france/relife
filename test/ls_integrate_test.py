import random
import pytest
import numpy as np
from relife.lifetime_model import ProportionalHazard, Weibull


M = 10
N = 3

@pytest.mark.parametrize(
    "model,a,b",
    [
        (
                Weibull(7, 0.05),
                random.uniform(2.5, 5),
                np.random.uniform(8, 10.0, size=(M,))
        ),
    ]
)
def test_ls_integrate_distri(model, a, b):
    assert model.ls_integrate(np.ones_like, a, b).shape == (M,)


NB_COEF = 2
@pytest.mark.parametrize(
    "model,a,b,covar",
    [
        (
                ProportionalHazard(Weibull(7, 0.05), coef=(random.uniform(1., 2.),)*NB_COEF),
                random.uniform(2.5, 5),
                np.random.uniform(8, 10.0, size=(M,N)),
                np.random.randn(M, NB_COEF),
        )
    ]
)
def test_ls_integrate_regression(model, a, b, covar):
    assert model.ls_integrate(np.ones_like, a, b, covar).shape == (M,N)