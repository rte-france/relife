.. currentmodule:: relife

User API
=========

.. rubric:: Parametric lifetime models

Lifetime models are objects whose exposed interface can answer basic probility functions of the survival analysis. In
object-oriented programming, one must not see them as a group of probability functions solely : it also encapsulates
data like parameters, other nested models, etc.

.. autosummary::
    :toctree: parametric_lifetime_models
    :template: default.rst
    :caption: Parametric lifetime models
    :nosignatures:

    models.Exponential
    models.Weibull
    models.Gompertz
    models.Gamma
    models.LogLogistic
    models.ProportionalHazard
    models.AFT


.. rubric:: Non-parametric lifetime models

Non-parametric lifetime estimators are objects used to compute well-knowns non-parametric estimations of some
probability functions. Their exposed interface follows the same logic of lifetime models except that they do not store
parameters but estimations of functions values.

.. autosummary::
    :toctree: nonparametric_models
    :template: default.rst
    :caption: Non-parametric lifetime models
    :nosignatures:

    models.ECDF
    models.KaplanMeier
    models.NelsonAalen
    models.Turnbull

.. rubric:: Other parametric models

.. autosummary::
    :toctree: parametric_models
    :template: default.rst
    :caption: Other parametric models
    :nosignatures:

    models.regression.CovarEffect


.. rubric:: Renewal policies

Renewal policies are objects that are generally composed of one underlying renewal process and one or more lifetime
model. Their interfaces expose a bunch of statiscal properties like expections and a sample procedure to generate
data.

.. autosummary::
    :toctree: renewal_policies
    :template: default.rst
    :caption: Renewal policies
    :nosignatures:

    policies.OneCycleAgeReplacementPolicy
    policies.OneCycleRunToFailure
    policies.RunToFailure
    policies.AgeReplacementPolicy