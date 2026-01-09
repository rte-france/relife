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

Lifetime models
---------------

Parametric lifetime models
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: relife.lifetime_model

.. rubric:: Lifetime distributions

.. autosummary::
    :toctree: parametric_lifetime_models
    :template: class_template.rst
    :nosignatures:

    Exponential
    Weibull
    Gompertz
    Gamma
    LogLogistic

.. rubric:: Lifetime regression

.. autosummary::
    :toctree: parametric_lifetime_models
    :template: class_template.rst
    :nosignatures:

    ProportionalHazard
    AcceleratedFailureTime

.. rubric:: Conditional lifetime models

.. autosummary::
    :toctree: parametric_lifetime_models
    :template: class_template.rst
    :nosignatures:

    LeftTruncatedModel
    AgeReplacementModel


Non parametric lifetime models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: non_parametric_lifetime_models
    :template: class_template.rst
    :caption: Non parametric lifetime models
    :nosignatures:

    KaplanMeier
    NelsonAalen
    ECDF

Stochastic processes
--------------------

.. currentmodule:: relife.stochastic_process

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

Convenient functions to instanciate a policy based on the given model and cost structure.

.. currentmodule:: relife.policy

.. autosummary::
    :toctree: policy
    :caption: Maintenance policies
    :template: function_template.rst
    :nosignatures:

    age_replacement_policy
    run_to_failure_policy

Lower-level policy classes that can be used in addition to the above functions.

.. autosummary::
    :toctree: policy
    :template: class_template.rst
    :nosignatures:

    AgeReplacementPolicy
    OneCycleAgeReplacementPolicy
    RunToFailurePolicy
    OneCycleRunToFailurePolicy
    NonHomogeneousPoissonAgeReplacementPolicy

Economy
-------

.. currentmodule:: relife.economic

.. autosummary::
    :toctree: economy
    :template: class_template.rst
    :caption: Economy
    :nosignatures:

    RunToFailureReward
    AgeReplacementReward
    ExponentialDiscounting

Built-in dataset
----------------

.. currentmodule:: relife.data

.. autosummary::
    :toctree: data
    :template: function_template.rst
    :caption: Built-in datasets
    :nosignatures:

    ~load_circuit_breaker
    ~load_insulator_string
    ~load_power_transformer

Utils
-----

Various utilities to help with development.

.. currentmodule:: relife.utils

.. autosummary::
    :toctree: routines
    :template: function_template.rst
    :caption: Utils
    :nosignatures:

    ~reshape_1d_arg
    ~get_args_nb_assets
    ~is_frozen
    ~is_lifetime_model
    ~is_non_homogeneous_poisson_process

Quadrature utilities used a many computations. We don't use Scipy quadrature implementations as, to our knowledge, they don't
support automatic broadcasting of 2D bounds.

.. currentmodule:: relife.utils.quadrature

.. autosummary::
    :toctree: routines
    :template: function_template.rst
    :nosignatures:

    ~legendre_quadrature
    ~laguerre_quadrature
    ~unweighted_laguerre_quadrature

