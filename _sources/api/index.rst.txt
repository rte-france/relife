API
===

This section provides comprehensive details about the exposed ReLife API.
ReLife is structured into different modules, each with a clear and specific role. We divided the API documentation close to the same logic.

Lifetime models
===============

.. currentmodule:: relife.lifetime_model

Lifetime models are objects whose exposed interface can answer basic probility functions of the survival analysis. In
object-oriented programming, one must not see them as a group of probability functions solely : it also encapsulates
data like parameters, other nested models, etc.

Lifetime models are imported like this :

.. code-block:: python

    from relife.lifetime_model import <model_constructor>

Here is an exhaustive list of all lifetime model constructors that can be used in Relife :

.. rubric:: Lifetime distributions

.. autosummary::
    :toctree: lifetime_distribution
    :template: default_template.rst
    :caption: Lifetime distributions
    :nosignatures:

    Exponential
    Weibull
    Gompertz
    Gamma
    LogLogistic

.. rubric:: Lifetime regression

.. autosummary::
    :toctree: lifetime_regression
    :template: default_template.rst
    :caption: Lifetime regressions
    :nosignatures:

    ProportionalHazard
    AcceleratedFailureTime

.. rubric:: Conditional lifetime models

.. autosummary::
    :toctree: conditional_model
    :template: default_template.rst
    :caption: Conditional lifetime models
    :nosignatures:

    LeftTruncatedModel
    AgeReplacementModel
