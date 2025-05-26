.. currentmodule:: relife

Lifetime models
===============

Lifetime models are objects whose exposed interface can answer basic probility functions of the survival analysis. In
object-oriented programming, one must not see them as a group of probability functions solely : it also encapsulates
data like parameters, other nested models, etc.

.. rubric:: Lifetime distributions

.. autosummary::
    :toctree: lifetime_distribution
    :template: default.rst
    :caption: Lifetime distributions
    :nosignatures:

    lifetime_model.Exponential
    lifetime_model.Weibull
    lifetime_model.Gompertz
    lifetime_model.Gamma
    lifetime_model.LogLogistic

.. rubric:: Lifetime regression

.. autosummary::
    :toctree: lifetime_regression
    :template: default.rst
    :caption: Lifetime regressions
    :nosignatures:

    lifetime_model.ProportionalHazard
    lifetime_model.AcceleratedFailureTime