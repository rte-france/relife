API
===

This section provides comprehensive details about the exposed ReLife API.
ReLife is structured into different modules, each with a clear and specific role. We divided the API documentation close to the same logic.

Lifetime models
---------------

Parametric lifetime models
^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    :toctree: parametric_lifetime_models
    :template: class_template.rst
    :caption: Parametric lifetime models
    :nosignatures:

    Exponential
    Weibull
    Gompertz
    Gamma
    LogLogistic

Here is a quick example that instanciates a ``Weibull`` distribution and computes the survival function.

.. code-block:: python

    >>> import numpy as np
    >>> from relife.lifetime_model import Weibull
    >>> weibull = Weibull(3.47, 0.01) # shape = 3.47, rate = 0.01
    # sf is np.float64, ie. a scalar
    >>> sf = weibull.sf(10.)
    # sf is np.array of shape (2,), 2 points for 1 asset
    >>> sf_1d = weibull.sf(np.array([10., 20.]))
    # sf is np.array of (2, 3), 3 points for 2 assets
    >>> sf_2d = weibull.sf(np.array([[10., 20., 30.], [30., 40., 50.]]))

.. rubric:: Lifetime regression

.. autosummary::
    :toctree: parametric_lifetime_models
    :template: class_template.rst
    :nosignatures:

    ProportionalHazard
    AcceleratedFailureTime

Here is a quick example that instanciates a ``ProportionalHazard`` regression from the same ``Weibull`` distribution (see above).
The regression has 3 coefficients.

.. code-block:: python

    >>> import numpy as np
    >>> from relife.lifetime_model import Weibull
    >>> from relife.lifetime_model import ProportionalHazard

    >>> weibull = Weibull(3.47, 0.01) # shape = 3.47, rate = 0.01
    # 3 coefficients
    >>> regression = ProportionalHazard(weibull, coefficients=(0.2, 0.01, 0.4))

    # 1 value per covar
    >>> covar = np.array([3., 59., 9.3])
    # sf is np.float64, ie. a scalar
    >>> sf = regression.sf(10., covar)

    # 2 values per covar, meaning two assets
    >>> covar_2d = np.array([[3., 59., 9.3], [2., 64., 5.6]])
    # sf is np.array of shape (2, 1), 1 point for 2 assets
    >>> sf_2d = regression.sf(10., covar_2d)

Note that the example above uses Numpy broadcasting. It is a core functionnality of ReLife. For more explanations and pratical
examples about broadcasting in ReLife, please read Broadcasting in ReLife.

.. rubric:: Conditional lifetime models

.. autosummary::
    :toctree: parametric_lifetime_models
    :template: class_template.rst
    :nosignatures:

    LeftTruncatedModel
    AgeReplacementModel

.. code-block:: python

    >>> import numpy as np
    >>> from relife.lifetime_model import LeftTruncatedModel, Weibull
    >>> from relife.lifetime_model import ProportionalHazard

    >>> left_truncated_weibull = LeftTruncated(Weibull(3.47, 0.01))

    # sf is np.float64, ie. a scalar
    >>> sf = left_truncated_weibull.sf(10., a0=20.)

.. rubric:: Frozen parametric lifetime models

.. currentmodule:: relife.lifetime_model

Frozen lifetime models share the same properties as lifetime models, but any additional arguments to time are frozen.
This means that the values of these arguments are stored within the model and cannot be set as arguments in a method request.
These models are important in ReLife because other objects expect frozen-like lifetime models: any lifetime model that has only time as a method argument.
It is specifically the case of policy objects. To freeze a lifetime model, do the following :

.. code-block:: python

    >>> from relife import freeze
    >>> frozen_model = freeze(<a_lifetime_model>, **kwargs)

``**kwargs`` is an unpacked dictionnary (a set of key-value pairs). Concretelly, a user must passed named arguments,
eg ``frozen_regression = freeze(regression, covar= ...)``. If the user passes an argument that is not appropriate with
the lifetime model, a error message will be raised.


.. note::

    ``LifetimeDistribution`` can't be frozen because it doesn't make sense (time is the only variable needed for these objects)


.. rubric:: Frozen lifetime regression

.. autosummary::
    :toctree: parametric_lifetime_models
    :template: class_template.rst
    :nosignatures:

    FrozenLifetimeRegression
    FrozenLeftTruncatedModel
    FrozenAgeReplacementModel


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
    Turnbull

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

Maintenance policies
--------------------


.. currentmodule:: relife.policy

.. autosummary::
    :toctree: policy
    :template: class_template.rst
    :caption: Maintenance policies
    :nosignatures:

    AgeReplacementPolicy
    OneCycleAgeReplacementPolicy
    RunToFailurePolicy
    OneCycleRunToFailurePolicy


Routines
--------

.. currentmodule:: relife

.. autosummary::
    :toctree: routines
    :template: function_template.rst
    :caption: Routines
    :nosignatures:

    ~freeze

Base classes
------------

.. warning::

    The interfaces presented below might interest you only if you want to understand how ReLife is implemented (contributions, suggestions, spotted errors, etc.)
    Otherwise, you can skip this part of the API

.. rubric:: Parametric models

.. currentmodule:: relife

.. autosummary::
    :toctree: base_class
    :template: class_template.rst
    :caption: Base classes
    :nosignatures:

    ParametricModel
    ~lifetime_model.ParametricLifetimeModel
    ~lifetime_model.FrozenParametricLifetimeModel
    ~lifetime_model.NonParametricLifetimeModel

.. rubric:: Policies

.. currentmodule:: relife.policy

.. autosummary::
    :toctree: base_class
    :template: class_template.rst
    :nosignatures:

    BaseAgeReplacementPolicy
    BaseOneCycleAgeReplacementPolicy