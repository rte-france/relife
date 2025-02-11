API
====

The content of the exposed ReLife API. It is where one can find details about object interfaces (how many parameters
are needeed by a function, what are their types, etc.).

.. currentmodule:: relife


Usual API
----------



.. rubric:: Lifetime models

Lifetime models are objects whose exposed interface can answer basic probility functions of the survival analysis. In
object-oriented programming, one must not see them as a group of probability functions solely : it also encapsulates
data like parameters, other nested models, etc.

.. autosummary::
    :toctree: lifetime_models
    :template: default.rst
    :caption: Lifetime models
    :nosignatures:

    models.Exponential
    models.Weibull
    models.Gompertz
    models.Gamma
    models.LogLogistic
    models.ProportionalHazard
    models.AFT


.. rubric:: Non-parametric lifetime estimators

Non-parametric lifetime estimators are objects used to compute well-knowns non-parametric estimations of some
probability functions. Their exposed interface follows the same logic of lifetime models except that they do not store
parameters but estimations of functions values.

.. autosummary::
    :toctree: nonparametric
    :template: default.rst
    :caption: Non-parametric lifetime estimators
    :nosignatures:

    models.ECDF
    models.KaplanMeier
    models.NelsonAalen
    models.Turnbull


.. rubric:: Renewal policies

Renewal policies are objects that are generally composed of one underlying renewal process and one or more lifetime
model. Their interfaces expose a bunch of statiscal properties like expections and a sample procedure to generate
data.

.. autosummary::
    :toctree: renewal_policy
    :template: default.rst
    :caption: Renewal policies
    :nosignatures:

    policies.OneCycleAgeReplacementPolicy
    policies.OneCycleRunToFailure
    policies.RunToFailure
    policies.AgeReplacementPolicy


Core objects
------------

.. rubric:: Likelihoods

.. autosummary::
    :toctree: likelihood
    :template: default.rst
    :caption: Likelihoods
    :nosignatures:

    core.likelihoods.LikelihoodFromLifetimes

.. rubric:: Nested models

A nested model is basically a `LifetimeModel` object that wraps another `LifetimeModel` object set as a baseline.
Those models mainly deserves to some functionnalities in `Policy` object but advanced users can clearly use them for a
specific purpose. For instance `LeftTruncatedModel` wraps a baseline model by overriding some of its probibability function
in order to take into account the time condition imposed by a left trunctation `a0`.

.. autosummary::
    :toctree: nested_models
    :template: parametric-lifetime-model-class.rst
    :caption: Nested models
    :nosignatures:

    core.nested_model.AgeReplacementModel
    core.nested_model.LeftTruncatedModel
    core.nested_model.EquilibriumDistribution

.. rubric:: Base model

What we call `base model` are objects (in Python class are objects) used at the core of ReLife subsystems.
For beginners, it is not necessary to know them. If you start to inspect ReLife code, you will encounter them regularly
in inheritance hierarchy. In fact, most objects created for fiability theory inherits from one of those objects. One can
think of them as `engines` that empower objects with special functionalities to make ReLife model creation easier.


.. autosummary::
    :toctree: base_class
    :template: default.rst
    :caption: Base class
    :nosignatures:

    core.model.Parameters
    core.model.ParametricModel
    core.model.LifetimeModel
    core.model.ParametricLifetimeModel
