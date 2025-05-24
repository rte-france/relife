from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass, field
from itertools import chain, filterfalse
from typing import TYPE_CHECKING, Any, Iterator, Optional, Self, Union, overload

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import OptimizeResult

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

    fitting_results: Optional[FittingResults]
    _nested_models: dict[str, ParametricModel]

    def __init__(self, **kwparams: Optional[float]):
        self._parameters = Parameters(**kwparams)
        self._nested_models = {}
        self.fitting_results = None

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
        return np.array(tuple(self._parameters.allvalues()))

    @params.setter
    def params(self, values: NDArray[np.float64]):
        if not isinstance(values, np.ndarray):
            raise ValueError(f"Incorrect params values. It must be contained in a 1d array. Got type {type(values)}")
        if values.ndim > 1:
            raise ValueError(f"Expected params values to be 1d array. Got {values.ndim} ndim")
        self._parameters.set_allvalues(*values)

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
        return tuple(self._parameters.allkeys())

    @property
    def nb_params(self) -> int:
        """
        Number of parameters.

        Returns
        -------
        int
            Number of parameters.

        """
        return len(self._parameters)

    def compose_with(self, **kwcomponents: Self):
        """Compose with new ``ParametricModel`` instance(s).

        This method must be seen as standard function composition exept that objects are not
        functions but group of functions (as object encapsulates functions). When you
        compose your ``ParametricModel`` instance with new one(s), the followings happen :

        - each new parameters are added to the current ``Parameters`` instance
        - each new `ParametricModel` instance is accessible as a standard attribute

        Like so, you can request new `ParametricModel` components in current `ParametricModel`
        instance while setting and getting all parameters. This is usefull when `ParametricModel`
        can be seen as a nested function (see `Regression`).

        Parameters
        ----------
        **kwcomponents : variadic named ``ParametricModel`` instance

            Instance names (keys) are followed by the instances themself (values).

        Notes
        -----
        If one wants to pass a `dict` of key-value, make sure to unpack the dict
        with `**` operator or you will get a nasty `TypeError`.
        """
        for name in kwcomponents.keys():
            if name in self._parameters.nodedata:
                raise ValueError(f"{name} already exists as param name")
            if name in self._nested_models:
                raise ValueError(f"{name} already exists as leaf function")
        for name, model in kwcomponents.items():
            self._nested_models[name] = model
            self._parameters.set_leaf(f"{name}.params", getattr(model, "_parameters"))

    def nested_models(self) -> Iterator[Self]:
        """parallel walk through key value pairs"""

        def items_walk(model: Self) -> Iterator:
            yield list(model._nested_models.items())
            for leaf in model._nested_models.values():
                yield list(chain.from_iterable(items_walk(leaf)))

        return chain.from_iterable(items_walk(self))

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("_parameters"):
            return super().__getattribute__("_parameters")[name]
        if name in super().__getattribute__("_nested_models"):
            return super().__getattribute__("_nested_models")[name]
        raise AttributeError(f"{class_name} has no attribute named {name}")

    def __setattr__(self, name: str, value: Any):
        if name in ("_parameters", "_nested_models"):
            super().__setattr__(name, value)
        elif name in self._parameters:
            self._parameters[name] = value
        elif name in self._nested_models:
            raise ValueError(
                "ParametricModel named {name} is already set. If you want to change it, recreate a ParametricModel"
            )
        elif isinstance(value, ParametricModel):
            self.compose_with(**{name: value})
        else:
            super().__setattr__(name, value)

    # def new_params(self, **kwparams: Optional[float]):
    #     """Change local parameters structure.
    #
    #     This method only affects **local** parameters. `ParametricModel` components are not
    #     affected. This is usefull when one wants to change core parameters for any reason. For
    #     instance `Regression` model use `new_params` to change number of regression coefficients
    #     depending on the number of covariates that are passed to the `fit` method.
    #
    #     Parameters
    #     ----------
    #     **kwparams : variadic named floats corresponding to new parameters
    #
    #         Float names (keys) are followed by float instances (values).
    #
    #     Notes
    #     -----
    #     If one wants to pass a `dict` of key-value, make sure to unpack the dict
    #     with `**` operator or you will get a nasty `TypeError`.
    #     """
    #
    #     for name in kwparams.keys():
    #         if name in self._nested_models.keys():
    #             raise ValueError(f"{name} already exists as function name")
    #     self._parameters.nodedata = kwparams


# TODO : MAYBE, recarrays (extended numpy structured arrays) can replace this huge Parameters class
# or see custom array containers : https://numpy.org/doc/stable/user/basics.dispatch.html#writing-custom-array-containers
class Parameters:
    """
    Dict-like tree structured parameters.

    Every ``ParametricModel`` are composed of a ``Parameters`` instance.
    """

    parent: Optional[Self]
    leaves: dict[str, Self]
    _nodedata: dict[str, float]  # np.nan is float
    _allkeys: tuple[str, ...]
    _allvalues: tuple[float, ...]

    def __init__(self, mapping: Optional[dict[str, Optional[float]]] = None, /, **kwargs: Optional[float]):
        self._nodedata = {}
        if mapping is not None:
            # not that np.nan is an instance of float
            self._nodedata = {k: v if v is not None else np.nan for k, v in mapping.items()}
        if kwargs:
            self._nodedata.update(**{k: v if v is not None else np.nan for k, v in kwargs.items()})
        self.parent = None
        self.leaves = {}
        self._allkeys = tuple(self._nodedata.keys())
        self._allvalues = tuple(self._nodedata.values())

    @property
    def nodedata(self) -> dict[str, float]:
        """data of current node as dict. Note that values can be np.nan which is instance of float"""
        return self._nodedata

    @nodedata.setter
    def nodedata(self, mapping: dict[str, Optional[float]]):
        self._nodedata = {k: np.nan if v is None else v for k, v in mapping.items()}
        self._update()

    def allkeys(self) -> Iterator[str]:
        """keys of current and leaf nodes as list"""
        return iter(self._allkeys)

    def allvalues(self) -> Iterator[float]:
        """values of current and leaf nodes as list"""
        return iter(self._allvalues)

    def allitems(self) -> Iterator[tuple[str, float]]:
        def items_walk(parameters: Self):
            yield tuple(parameters._nodedata.items())
            for leaf in parameters.leaves.values():
                yield tuple(chain.from_iterable(items_walk(leaf)))

        return chain.from_iterable(items_walk(self))

    def set_allvalues(self, *values: Optional[float]):
        if len(values) != len(self):
            raise ValueError(f"values expects {len(self)} items but got {len(values)}")
        self._allvalues = tuple((np.nan if v is None else v for v in values))
        pos = len(self._nodedata)
        self._nodedata.update(zip(self._nodedata.keys(), (np.nan if v is None else v for v in values[:pos])))
        for leaf in self.leaves.values():
            leaf.set_allvalues(*values[pos : pos + len(leaf)])
            pos += len(leaf)
        if self.parent is not None:
            self.parent._update()

    def set_allkeys(self, *keys: str):
        if len(keys) != len(self):
            raise ValueError(f"names expects {len(self)} items but got {len(keys)}")
        self._allkeys = keys
        pos = len(self._nodedata)
        self._nodedata = {keys[:pos][i]: v for i, v in enumerate(self._nodedata.values())}
        for leaf in self.leaves.values():
            leaf.set_allkeys(*keys[pos : pos + len(leaf)])
            pos += len(leaf)
        if self.parent is not None:
            self.parent._update()

    def __len__(self) -> int:
        return len(self._allkeys)

    def __contains__(self, item: str):
        """contains only applies on current node"""
        return item in self._nodedata

    def __getitem__(self, item: str):
        return self._nodedata[item]

    def __setitem__(self, key: str, value: Optional[float]):
        self._nodedata[key] = value if value is not None else np.nan
        self._update()

    def __delitem__(self, key: str):
        del self._nodedata[key]
        self._update()

    def get_leaf(self, item):
        return self.leaves[item]

    def set_leaf(self, key, value):
        if key not in self.leaves:
            value.parent = self
        self.leaves[key] = value
        self._update()

    def del_leaf(self, key):
        del self.leaves[key]
        self._update()

    def _update(self):
        """update names and values of current and parent nodes"""
        try:
            next(self.allitems())
            _k, _v = zip(*self.allitems())
            self._allkeys = tuple(_k)
            self._allvalues = tuple(_v)
        except StopIteration:
            pass

        if self.parent is not None:
            self.parent._update()


@dataclass
class FittingResults:
    """Fitting results of the parametric_model core."""

    nb_samples: InitVar[int]  #: Number of observations (samples).

    opt: InitVar[OptimizeResult] = field(repr=False)  #: Optimization result (see scipy.optimize.OptimizeResult doc).
    var: Optional[NDArray[np.float64]] = field(
        repr=False, default=None
    )  #: Covariance matrix (computed as the inverse of the Hessian matrix).
    params: NDArray[np.float64] = field(init=False)  #: Optimal parameters values
    nb_params: int = field(init=False, repr=False)  #: Number of parameters.
    AIC: float = field(init=False)  #: Akaike Information Criterion.
    AICc: float = field(init=False)  #: Akaike Information Criterion with a correction for small sample sizes.
    BIC: float = field(init=False)  #: Bayesian Information Criterion.
    se: Optional[NDArray[np.float64]] = field(
        init=False, repr=False
    )  #: Standard error, square root of the diagonal of the covariance matrix
    IC: Optional[NDArray[np.float64]] = field(init=False, repr=False)  #: 95% IC

    def __post_init__(self, nb_samples, opt):
        self.params = opt.x  # (p,)
        self.nb_params = opt.x.size
        self.AIC = float(2 * self.nb_params + 2 * opt.fun)
        self.AICc = float(self.AIC + 2 * self.nb_params * (self.nb_params + 1) / (nb_samples - self.nb_params - 1))
        self.BIC = float(np.log(nb_samples) * self.nb_params + 2 * opt.fun)

        self.se = None
        if self.var is not None:
            self.se = np.sqrt(np.diag(self.var))
            self.IC = self.params.reshape(-1, 1) + stats.norm.ppf((0.05, 0.95)) * self.se / np.sqrt(
                nb_samples
            )  # (p, 2)

        # TODO : ajouter IC95% et tirer 100 tirage et verifier si parametre dans l'intervalle

    def se_estimation_function(self, jac_f: np.ndarray) -> np.float64 | NDArray[np.float64]:
        """Standard error estimation function.

        Parameters
        ----------
        jac_f : 1D array
            The Jacobian of a function f with respect to params.

        Returns
        -------
        1D array
            Standard error for f(params).

        References
        ----------
        .. [1] Meeker, W. Q., Escobar, L. A., & Pascual, F. G. (2022).
            Statistical methods for reliability data. John Wiley & Sons.
        """
        # [1] equation B.10 in Appendix
        # jac_f : (p,), (p, n) or (p, m, n)
        # self.var : (p, p)
        if self.var is not None:
            return np.sqrt(np.einsum("p...,pp,p...", jac_f, self.var, jac_f))  # (), (n,) or (m, n)
        raise ValueError("Can't compute if var is None")

    def asdict(self) -> dict:
        """converts FittingResult into a dictionary.

        Returns
        -------
        dict
            Returns the fitting result as a dictionary.
        """
        return asdict(self)

    def __str__(self) -> str:
        """Returns a string representation of FittingResults with fields in a single column."""
        fields = [("fitted params", self.params), ("AIC", self.AIC), ("AICc", self.AICc), ("BIC", self.BIC)]
        # Find the maximum field name length for alignment
        max_name_length = max(len(name) for name, _ in fields)
        lines = []
        for name, value in fields:
            # Format arrays to be more compact
            if isinstance(value, np.ndarray):
                value_str = f"[{', '.join(f'{x:.6g}' for x in value)}]"
            else:
                value_str = f"{value:.6g}" if isinstance(value, float) else str(value)
            lines.append(f"{name:<{max_name_length}} : {value_str}")
        return "\n".join(lines)


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

    if not hasattr(model, "args_names"):
        raise ValueError
    args_names = model.args_names
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
        LifetimeDistribution,
        FrozenLifetimeRegression,
        FrozenLeftTruncatedModel,
        FrozenAgeReplacementModel,
        FrozenNonHomogeneousPoissonProcess,
    ],
) -> int:

    from relife.lifetime_model import (
        FrozenAgeReplacementModel,
        FrozenLeftTruncatedModel,
        FrozenLifetimeRegression,
        LifetimeDistribution,
    )
    from relife.stochastic_process import FrozenNonHomogeneousPoissonProcess

    model_chain: (
        ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
        | FrozenParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
    )
    nb_assets_list: list[int] = []
    if isinstance(model, FrozenNonHomogeneousPoissonProcess):
        model_chain = model.unfreeze().baseline
    else:
        model_chain = model
    if isinstance(model_chain, LifetimeDistribution):
        return 1

    for arg in model.frozen_args:
        if isinstance(model_chain, FrozenLifetimeRegression):
            # arg is covar
            nb_assets_list.append(np.atleast_2d(np.asarray(arg)).shape[0])
            model_chain = model_chain.unfreeze().baseline
        elif isinstance(model_chain, LifetimeRegression):
            # arg is covar
            nb_assets_list.append(np.atleast_2d(np.asarray(arg)).shape[0])
            model_chain = model_chain.baseline
        elif isinstance(model_chain, FrozenLeftTruncatedModel):
            # arg is a0
            nb_assets_list.append(np.asarray(arg).size)
            model_chain = model_chain.baseline
        elif isinstance(model_chain, FrozenAgeReplacementModel):
            # arg is ar
            nb_assets_list.append(np.asarray(arg).size)
            model_chain = model_chain.baseline

    non_one_nb_assets = set(filterfalse(lambda x: x == 1, nb_assets_list))
    if len(non_one_nb_assets) > 1:
        raise ValueError(f"Invalid number of assets given in arguments. Got several nb assets : {non_one_nb_assets}")
    return list(non_one_nb_assets)[0]


# see sklearn/base.py : return unfitted ParametricModel
# def clone(model: ParametricModel) -> ParametricModel: ...
