"""
``relife.lifetime_model``
=========================

The ReLife lifetime_model module exposes various stochastic models to modelize
lifetime data. Internal operations are computed using NumPy and Scipy.

- NumPy: https://github.com/numpy/numpy
- Scipy: https://github.com/scipy/scipy

Objects present in relife.lifetime_model are listed below.

Lifetime distributions
----------------------

    Exponential
    Weibull
    Gompertz
    Gamma
    LogLogistic
    MinimumDistribution
    EquilibriumDistribution


Lifetime regressions
--------------------

    ParametricProportionalHazard
    ParametricAcceleratedFailureTime

Semiparametric lifetime regression
----------------------------------

    SemiParametricProportionalHazard


Nonparametric models
--------------------

    KaplanMeier
    ECDF
    NelsonAalen

Likelihoods
-----------

    LifetimeLikelihood
    CoxPartialLifetimeLikelihood
    BreslowPartialLifetimeLikelihood
    EfronPartialLifetimeLikelihood

"""

from . import (
    _base,
    _distributions,
    _non_parametric_models,
    _parametric_regressions,
    _semi_parametric_regressions,
)

_non_parametric_api = [
    "KaplanMeier",
    "ECDF",
    "NelsonAalen",
]

__all__: list[str] = []
# __all__ = [
#     "CoxPartialLifetimeLikelihood",
#     "EfronPartialLifetimeLikelihood",
#     "BreslowPartialLifetimeLikelihood",
# ]
__all__ += _base.__all__
__all__ += _distributions.__all__
__all__ += _parametric_regressions.__all__
__all__ += _non_parametric_models.__all__
__all__ += _semi_parametric_regressions.__all__
