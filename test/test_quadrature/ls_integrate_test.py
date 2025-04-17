import random
import pytest
import numpy as np
from relife.lifetime_model import ProportionalHazard, Weibull, Gompertz, Gamma, LogLogistic, AFT

M = 10
N = 3
NB_COEF = 2
random.seed(10)
np.random.seed(10)


lifetime_distributions = [
    Weibull(2, 0.05),
    Gompertz(0.01, 0.1),
    Gamma(2, 0.05),
    LogLogistic(3, 0.05),
]

lifetime_regressions = [ProportionalHazard(distri, coef=(random.uniform(1.0, 2.0),) * NB_COEF) for distri in lifetime_distributions]
lifetime_regressions += [AFT(distri, coef=(random.uniform(1.0, 2.0),) * NB_COEF) for distri in lifetime_distributions]



AB_1D_DISTRI = [
    (
        random.uniform(2.5, 5.),
        np.random.uniform(8., 10.0, size=(M,))
    ),
    (
        np.random.uniform(2.5, 5., size=(M,)),
        random.uniform(8., 10.),
    )
]

AB_2D_DISTRI = [
    (
        np.random.uniform(2.5, 5., size=(M, N)),
        np.random.uniform(8., 10.0, size=(M, N))
    ),
    (
        np.random.uniform(2.5, 5., size=(M, 1)),
        np.random.uniform(8., 10.0, size=(1, N))
    ),
    (
        np.random.uniform(2.5, 5., size=(1, N)),
        np.random.uniform(8., 10.0, size=(M, 1))
    ),
    (
        random.uniform(2.5, 5.),
        np.random.uniform(8., 10.0, size=(M, N))
    ),
    (
        np.random.uniform(2.5, 5., size=(M, N)),
        random.uniform(8., 10.),
    ),
    (
        np.random.uniform(2.5, 5., size=(M,)),
        np.random.uniform(8., 10.0, size=(M, N))
    ),
    (
        np.random.uniform(2.5, 5., size=(M, N)),
        np.random.uniform(8., 10., size=(M,))
    )

]




@pytest.mark.parametrize(
    "model,a,b",
    [
        (
                Weibull(7, 0.05),
                random.uniform(2.5, 5.),
                np.random.uniform(8., 10.0, size=(M,))
        ),
        (
                Weibull(7, 0.05),
                np.random.uniform(2.5, 5., size=(M,)),
                random.uniform(8., 10.),

        ),

    ]
)
def test_ls_integrate_distri(model, a, b):
    assert model.ls_integrate(np.ones_like, a, b).shape == (M,)


@pytest.mark.parametrize(
    "model,a,b",
    [
        (
                Weibull(7, 0.05),
                np.random.uniform(2.5, 5., size=(M,N)),
                np.random.uniform(8., 10.0, size=(M,N))
        ),
        (
                Weibull(7, 0.05),
                np.random.uniform(2.5, 5., size=(M, 1)),
                np.random.uniform(8., 10.0, size=(1, N))
        ),
        (
                Weibull(7, 0.05),
                np.random.uniform(2.5, 5., size=(1, N)),
                np.random.uniform(8., 10.0, size=(M, 1))
        ),
        (
                Weibull(7, 0.05),
                random.uniform(2.5, 5.),
                np.random.uniform(8., 10.0, size=(M, N))
        ),
        (
                Weibull(7, 0.05),
                np.random.uniform(2.5, 5., size=(M,N)),
                random.uniform(8., 10.),
        ),
        (
                Weibull(7, 0.05),
                np.random.uniform(2.5, 5., size=(M,)),
                np.random.uniform(8., 10.0, size=(M, N))
        ),
        (
                Weibull(7, 0.05),
                np.random.uniform(2.5, 5., size=(M, N)),
                np.random.uniform(8., 10., size=(M,))
        ),

    ]
)
def test_ls_integrate_distri_2d(model, a, b):
    assert model.ls_integrate(np.ones_like, a, b).shape == (M,N)





@pytest.mark.parametrize(
    "model,a,b,covar",
    [
        (
                ProportionalHazard(Weibull(7, 0.05), coef=(random.uniform(1., 2.),)*NB_COEF),
                random.uniform(2.5, 5),
                np.random.uniform(8, 10.0, size=(M,)),
                np.random.randn(M, NB_COEF),
        ),
        (
                ProportionalHazard(Weibull(7, 0.05), coef=(random.uniform(1., 2.),) * NB_COEF),
                random.uniform(2.5, 5),
                np.random.uniform(8, 10.0, size=(M,)),
                np.random.randn(M, NB_COEF),
        )
    ]
)
def test_ls_integrate_regression(model, a, b, covar):
    assert model.ls_integrate(np.ones_like, a, b, covar).shape == (M,)






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
def test_ls_integrate_regression_2d(model, a, b, covar):
    assert model.ls_integrate(np.ones_like, a, b, covar).shape == (M,N)