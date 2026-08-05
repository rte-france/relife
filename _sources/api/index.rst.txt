API
===

This section provides comprehensive details about the exposed ReLife API.
ReLife is structured into different modules, each with a clear and specific role.
We divided the API documentation close to the same logic.

.. currentmodule:: relife.base

.. autosummary::
    :toctree: base_class
    :template: class_template.rst
    :caption: Base classes
    :nosignatures:

    ParametricModel
    MaximumLikelihoodOptimizer

Lifetime models
---------------

Parametric lifetime models
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: relife.lifetime_models

.. rubric:: Parametric lifetime distributions

.. autosummary::
    :toctree: parametric_lifetime_models
    :template: class_template.rst
    :caption: Parametric lifetime models
    :nosignatures:

    Exponential
    Weibull
    Gompertz
    Gamma
    LogLogistic
    EquilibriumDistribution
    MinimumDistribution

.. rubric:: Parametric lifetime regressions

.. autosummary::
    :toctree: parametric_lifetime_models
    :template: class_template.rst
    :nosignatures:

    ParametricProportionalHazard
    ParametricAcceleratedFailureTime


Semiparametric lifetime regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: semiparametric_lifetime_models
    :template: class_template.rst
    :caption: Semiparametric lifetime models
    :nosignatures:

    SemiParametricProportionalHazard


Non parametric lifetime models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: nonparametric_lifetime_models
    :template: class_template.rst
    :caption: Nonparametric lifetime models
    :nosignatures:

    KaplanMeier
    NelsonAalen
    ECDF


Stochastic processes
--------------------

.. currentmodule:: relife.stochastic_processes

.. autosummary::
    :toctree: stochastic_process
    :template: class_template.rst
    :caption: Stochastic processes
    :nosignatures:

    RenewalProcess
    RenewalRewardProcess
    NonHomogeneousPoissonProcess

Maintenance policies
--------------------

.. currentmodule:: relife.policies

.. autosummary::
    :toctree: maintenance_policies
    :caption: Maintenance policies
    :template: class_template.rst
    :nosignatures:

    AgeReplacementPolicy
    OneCycleAgeReplacementPolicy
    RunToFailurePolicy
    OneCycleRunToFailurePolicy
    NonHomogeneousPoissonAgeReplacementPolicy

Built-in datasets
-----------------

.. currentmodule:: relife.datasets

.. autosummary::
    :toctree: datasets
    :template: function_template.rst
    :caption: Built-in datasets
    :nosignatures:

    ~load_circuit_breaker
    ~load_insulator_string
    ~load_power_transformer


Quadratures
-----------

Quadratures are used a many computations. We don't use Scipy quadrature implementations as, to our knowledge, they don't
support automatic broadcasting of 2D bounds.

.. currentmodule:: relife.quadratures

.. autosummary::
    :toctree: quadratures
    :template: function_template.rst
    :caption: Quadratures
    :nosignatures:

    ~legendre_quadrature
    ~laguerre_quadrature
    ~unweighted_laguerre_quadrature


Typing
------

.. currentmodule:: relife.typing

.. autosummary::
    :toctree: typing
    :template: data_template.rst
    :caption: Typing
    :nosignatures:

    ~CovarTs
    ~CoercibleFloat64_ND
    ~CoercibleFloat64_1D
    ~Float64_ND
    ~Float64_1D
    ~Timeline
    ~Seed


