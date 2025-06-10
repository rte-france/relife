from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, Optional, Self, Union, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from relife.lifetime_model import (
        AgeReplacementModel,
        FrozenAgeReplacementModel,
        FrozenLeftTruncatedModel,
        FrozenLifetimeRegression,
        FrozenParametricLifetimeModel,
        LeftTruncatedModel,
        LifetimeDistribution,
        LifetimeRegression,
        ParametricLifetimeModel,
    )
    from relife.stochastic_process import (
        FrozenNonHomogeneousPoissonProcess,
        NonHomogeneousPoissonProcess,
    )


class ParametricModel:
    """
    Base class to create a parametric_model core.

    Any parametric_model core must inherit from `ParametricModel`.
    """

    _nested_models: dict[str, ParametricModel]

    def __init__(self, **kwparams: Optional[float]):
        self._params = Parameters(**kwparams)
        self._nested_models = {}

    @property
    def params(self) -> NDArray[np.float64]:
        """
        Parameters values.

        Returns
        -------
        ndarray
            Parameters values of the core

        Notes
        -----
        If parameter values are not set, they are encoded as `np.nan` value.

        Parameters can be by manually setting`params` through its setter, fitting the core if `fit` exists or
        by specifying all parameters values when the core object is initialized.
        """
        return np.array(self._params.get_values())

    @params.setter
    def params(self, new_params: NDArray[np.float64]):
        if not isinstance(new_params, np.ndarray):
            raise ValueError(
                f"Incorrect params values. It must be contained in a 1d array. Got type {type(new_params)}"
            )
        if new_params.ndim > 1:
            raise ValueError(f"Expected params values to be 1d array. Got {new_params.ndim} ndim")
        self._params.set_values(tuple(v.item() for v in new_params))

    @property
    def params_names(self) -> tuple[str, ...]:
        """
        Parameters names.

        Returns
        -------
        list of str
            Parameters names

        Notes
        -----
        Parameters values can be requested (a.k.a. get) by their name at instance level.
        """
        return self._params.get_names()

    @property
    def nb_params(self) -> int:
        """
        Number of parameters.

        Returns
        -------
        int
            Number of parameters.

        """
        return self._params.size

    def _set_nested_model(self, name: str, model: ParametricModel):
        """Compose with new ``ParametricModel`` instance(s).

        This method acts like function. When you set a nested model instance, the followings happen :
        - each new parameters are added to the current ``Parameters`` instance
        - each new `ParametricModel` instance is accessible as a standard attribute

        Parameters
        ----------
        name : str
            model name
        model : ParametricModel
            model instance
        """
        self._nested_models[name] = model
        self._params.set_leaf(f"{name}.params", getattr(model, "_params"))

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("_params").get_names():
            return super().__getattribute__("_params").get_param_value(name)
        if name in super().__getattribute__("_nested_models"):
            return super().__getattribute__("_nested_models").get(name)
        raise AttributeError(f"{class_name} has no attribute named {name}")

    def __setattr__(self, name: str, value: Any):
        if name in ("_params", "_nested_models"):
            super().__setattr__(name, value)
        elif name in self._params.get_names():
            self._params.set_param_value(name, value)
        elif isinstance(value, ParametricModel):
            self._set_nested_model(name, value)
        else:  # just set a new attribute
            super().__setattr__(name, value)


# MAYBE, custom array container can replace it : https://numpy.org/doc/stable/user/basics.dispatch.html#writing-custom-array-containers
class Parameters:
    """
    Dict-like tree structured parameters.

    Every ``ParametricModel`` are composed of a ``Parameters`` instance.
    """

    _parent: Optional[Self]
    _leaves: dict[str, Self]
    _nodemapping: dict[str, float]  # np.nan is float
    _names: tuple[str, ...]
    _values: tuple[float, ...]

    def __init__(self, **kwargs: Optional[float]):
        self._parent = None
        self._leaves = {}
        self._nodemapping = {}
        self._values = ()
        self._names = ()
        if bool(kwargs):
            self.set_node(kwargs)

    def get_names(self) -> tuple[str, ...]:
        return self._names

    def get_values(self) -> tuple[float, ...]:
        return self._values

    def get_leaf(self, name: str) -> Parameters:
        try:
            return self._leaves[name]
        except KeyError:
            raise ValueError(f"Parameters object does not have leaf parameters called {name} in its scope")

    def get_param_value(self, name: str) -> float:
        try:
            return self._nodemapping[name]
        except KeyError:
            raise ValueError(f"Parameters object does not have parameter name called {name} in its scope")

    def set_param_value(self, name: str, value: float) -> None:
        if name not in self._nodemapping:
            raise ValueError(f"Parameters object does not have parameter name called {name} in its scope")
        self._nodemapping[name] = value
        self._update_names_and_values()

    @property
    def size(self) -> int:
        return len(self._values)

    def set_node(self, mapping: dict[str, Optional[float]]) -> None:
        """
        set node dict
        """
        self._nodemapping = {k: v if v is not None else np.nan for k, v in mapping.items()}
        self._update_names_and_values()  # update _names and _values

    def set_leaf(self, leaf_name: str, leaf: Parameters) -> None:
        """
        set a leaf or new leaf
        """
        if leaf_name not in self._leaves:
            leaf._parent = self
        self._leaves[leaf_name] = leaf
        self._update_names_and_values()  # update _names and _values

    def set_values(self, values: tuple[float, ...]) -> None:
        """set values of all tree"""
        if len(values) != self.size:
            raise ValueError(f"Expected {self.size} values but got {len(values)}")
        pos = len(self._nodemapping)
        self._nodemapping.update(zip(self._nodemapping.keys(), (np.nan if v is None else v for v in values[:pos])))
        self._values = tuple((np.nan if v is None else v for v in values))
        for leaf in self._leaves.values():
            leaf.set_values(values[pos : pos + leaf.size])
            pos += leaf.size
        if self._parent is not None:
            self._parent._update_names_and_values()

    def _update_names_and_values(self) -> None:
        """update names and values of current and parent nodes"""

        def items_walk(parameters: Self):
            yield tuple(parameters._nodemapping.items())
            for leaf in parameters._leaves.values():
                yield tuple(chain.from_iterable(items_walk(leaf)))

        generator = chain.from_iterable(items_walk(self))
        try:
            self._names, self._values = zip(*generator)
        except StopIteration:
            pass
        # recursively update parent
        if self._parent is not None:
            self._parent._update_names_and_values()


def model_args_names(
    model: Union[
        LifetimeDistribution, LifetimeRegression, LeftTruncatedModel, AgeReplacementModel, NonHomogeneousPoissonProcess
    ],
) -> tuple[str, ...]:
    from relife.lifetime_model import (
        AgeReplacementModel,
        FrozenParametricLifetimeModel,
        LeftTruncatedModel,
        LifetimeDistribution,
        LifetimeRegression,
    )
    from relife.stochastic_process import (
        NonHomogeneousPoissonProcess,
    )

    args_names = ()
    if isinstance(model, LifetimeRegression):
        args_names = args_names + ("covar",) + model_args_names(model.baseline)
    elif isinstance(model, AgeReplacementModel):
        args_names = args_names + ("ar",) + model_args_names(model.baseline)
    elif isinstance(model, LeftTruncatedModel):
        args_names = args_names + ("a0",) + model_args_names(model.baseline)
    elif isinstance(model, NonHomogeneousPoissonProcess):
        args_names = model_args_names(model.baseline)
    elif isinstance(model, (LifetimeDistribution, FrozenParametricLifetimeModel)):
        return args_names
    else:
        raise ValueError
    return args_names


@overload
def freeze(model: LifetimeRegression, **kwargs: float | NDArray[np.float64]) -> FrozenLifetimeRegression: ...


@overload
def freeze(model: LeftTruncatedModel, **kwargs: float | NDArray[np.float64]) -> FrozenLeftTruncatedModel: ...


@overload
def freeze(model: AgeReplacementModel, **kwargs: float | NDArray[np.float64]) -> FrozenAgeReplacementModel: ...


@overload
def freeze(
    model: NonHomogeneousPoissonProcess, **kwargs: float | NDArray[np.float64]
) -> FrozenNonHomogeneousPoissonProcess: ...


def freeze(
    model: LifetimeRegression | LeftTruncatedModel | AgeReplacementModel | NonHomogeneousPoissonProcess,
    **kwargs: float | NDArray[np.float64],
) -> (
    FrozenLifetimeRegression | FrozenLeftTruncatedModel | FrozenAgeReplacementModel | FrozenNonHomogeneousPoissonProcess
):
    """
    Freeze a parametric model with given arguments.

    This function takes a parametric model instance (lifetime model or stochastic process) and stores the given arguments
    in a new model interface allowing to resquest functions only with ``time`` argument.

    Parameters
    ----------
    model: LifetimeRegression, LeftTruncatedModel, AgeReplacementModel or NonHomogeneousPoissonProcess
        The model to be frozen. It should be one of the supported model types.
    **kwargs: float or np.ndarray
        Keyword arguments representing the parameter names and their respective values to be frozen.

    Returns
    -------
    FrozenLifetimeRegression, FrozenLeftTruncatedModel, FrozenAgeReplacementModel or FrozenNonHomogeneousPoissonProcess
        The frozen version of the input model with immutable provided argument values.

    Raises
    ------
    ValueError
        Raised when the provided arguments do not match the expected argument names
        for the given model, or if the model is not one of the supported types.


    Examples
    --------
    >>> from relife import freeze
    >>> from relife.lifetime_model import Weibull, ProportionalHazard
    >>> weibull = Weibull(3.5, 0.01)
    >>> regression = ProportionalHazard(weibull, coefficients=(1., 2.))
    >>> frozen_regression = freeze(regression, covar=1.5)
    >>> sf = frozen_regression.sf(np.array([10, 20])) # covar is fixed at 1.5


    """

    from relife.lifetime_model import (
        AgeReplacementModel,
        FrozenAgeReplacementModel,
        FrozenLeftTruncatedModel,
        FrozenLifetimeRegression,
        LeftTruncatedModel,
        LifetimeRegression,
    )
    from relife.stochastic_process import (
        FrozenNonHomogeneousPoissonProcess,
        NonHomogeneousPoissonProcess,
    )

    args_names = model_args_names(model)
    set_args_names = set(args_names)
    set_keys = set(kwargs.keys())
    if set_args_names != set_keys:
        raise ValueError(f"Expected {set_args_names} but got {set_keys}")

    position_mapping = {name: i for i, name in enumerate(args_names)}
    reorder_kwargs = dict(sorted(kwargs.items(), key=lambda item: position_mapping[item[0]]))
    args_values: tuple[float | NDArray[np.float64], ...] = tuple(reorder_kwargs.values())

    frozen_model: Union[
        FrozenLifetimeRegression,
        FrozenLeftTruncatedModel,
        FrozenAgeReplacementModel,
        FrozenNonHomogeneousPoissonProcess,
    ]
    # here args_nb_assets is set to 1 by default and then, will be overriden after testing frozen_args coherence
    if isinstance(model, LifetimeRegression):
        frozen_model = FrozenLifetimeRegression(model, 1, args_values[0], *args_values[1:])
    elif isinstance(model, AgeReplacementModel):
        frozen_model = FrozenAgeReplacementModel(model, 1, args_values[0], *args_values[1:])
    elif isinstance(model, LeftTruncatedModel):
        frozen_model = FrozenLeftTruncatedModel(model, 1, args_values[0], *args_values[1:])
    elif isinstance(model, NonHomogeneousPoissonProcess):
        frozen_model = FrozenNonHomogeneousPoissonProcess(model, 1, *args_values)
    else:
        raise ValueError
    frozen_model.args_nb_assets = get_args_nb_assets(frozen_model)
    return frozen_model


def get_args_nb_assets(
    model: Union[
        FrozenLifetimeRegression,
        FrozenLeftTruncatedModel,
        FrozenAgeReplacementModel,
        FrozenNonHomogeneousPoissonProcess,
    ],
) -> int:

    from relife.lifetime_model import (
        AgeReplacementModel,
        FrozenAgeReplacementModel,
        FrozenLeftTruncatedModel,
        FrozenLifetimeRegression,
        LeftTruncatedModel,
        LifetimeRegression,
    )
    from relife.stochastic_process import FrozenNonHomogeneousPoissonProcess

    local_model: (
        ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
        | FrozenParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
    )
    arg_nb_assets: int = 1
    arg_name: Optional[str] = None
    if isinstance(model, FrozenNonHomogeneousPoissonProcess):
        local_model = model.unfreeze().baseline
    else:
        local_model = model

    _arg_nb_assets: int = 1
    for arg in model.frozen_args:
        if isinstance(local_model, FrozenLifetimeRegression):
            # arg is covar
            _arg_nb_assets = np.atleast_2d(np.asarray(arg)).shape[0]
            arg_name = "covar"
            local_model = local_model.unfreeze().baseline
        elif isinstance(local_model, LifetimeRegression):
            # arg is covar
            _arg_nb_assets = np.atleast_2d(np.asarray(arg)).shape[0]
            arg_name = "covar"
            local_model = local_model.baseline
        elif isinstance(local_model, FrozenLeftTruncatedModel):
            # arg is a0
            _arg_nb_assets = np.asarray(arg).size
            arg_name = "a0"
            local_model = local_model.unfreeze().baseline
        elif isinstance(local_model, LeftTruncatedModel):
            # arg is a0
            _arg_nb_assets = np.asarray(arg).size
            arg_name = "a0"
            local_model = local_model.baseline
        elif isinstance(local_model, FrozenAgeReplacementModel):
            # arg is ar
            _arg_nb_assets = np.asarray(arg).size
            arg_name = "ar"
            local_model = local_model.unfreeze().baseline
        elif isinstance(local_model, AgeReplacementModel):
            # arg is ar
            _arg_nb_assets = np.asarray(arg).size
            arg_name = "ar"
            local_model = local_model.baseline
        if arg_name is not None:
            # test if nb_assets changed and would not be broadcastable
            if _arg_nb_assets != 1 and arg_nb_assets != 1 and _arg_nb_assets != arg_nb_assets:
                raise ValueError(
                    f"Invalid number of assets given in arguments. Got several nb assets. {arg_name} has {_arg_nb_assets} but already got {arg_nb_assets}"
                )
            arg_nb_assets = _arg_nb_assets
    return arg_nb_assets


# see sklearn/base.py : return unfitted ParametricModel
# def clone(model: ParametricModel) -> ParametricModel: ...
