"""Base classes for all parametric models."""

from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import chain
from typing import (
    Any,
    Generic,
    Literal,
    Self,
    TypeVar,
    TypeVarTuple,
    final,
    overload,
)

import numpy as np
from optype.numpy import Array, Array1D, Array2D, ToFloat, ToFloat1D
from scipy import stats
from scipy.optimize import approx_fprime, minimize
from typing_extensions import override

__all__ = ["ParametricModel", "FrozenParametricModel", "MaximumLikelihoodOptimizer"]


@final
class _Parameters:
    """
    Dict-like tree structured of parameters.

    Every ``ParametricModel`` are composed of a ``_Parameters`` instance.
    """

    parent: Self | None
    _leaves: dict[str, _Parameters]
    _mapping: dict[str, float]
    _all_values: tuple[float, ...]
    _all_names: tuple[str, ...]

    def __init__(self, **kwargs: float | None) -> None:
        self.parent = None
        self._leaves = {}
        self._mapping = {}
        self._all_values = ()
        self._all_names = ()
        if bool(kwargs):
            self._mapping = {
                k: v if v is not None else np.nan for k, v in kwargs.items()
            }
            self.update_tree()  # update _names and _values

    @property
    def all_names(self) -> tuple[str, ...]:
        return self._all_names

    @property
    def all_values(self) -> tuple[float, ...]:
        return self._all_values

    @property
    def size(self) -> int:
        return len(self.all_values)

    def set_leaf(self, leaf_name: str, leaf: Self) -> None:
        """
        set a leaf or new leaf
        """
        if leaf_name not in self._leaves:
            leaf.parent = self
        self._leaves[leaf_name] = leaf
        self.update_tree()  # update _names and _values

    def set_all_values(self, values: tuple[float | None, ...]) -> None:
        """set values of all tree"""
        if len(values) != self.size:
            raise ValueError(f"Expected {self.size} values but got {len(values)}")
        pos = len(self._mapping.items())
        self._mapping.update(
            zip(
                self._mapping.keys(),
                (np.nan if v is None else v for v in values[:pos]),
                strict=False,
            )
        )
        self._all_values = tuple(np.nan if v is None else v for v in values)
        for leaf in self._leaves.values():
            leaf.set_all_values(values[pos : pos + leaf.size])
            pos += leaf.size
        if self.parent is not None:
            self.parent.update_tree()

    def __getitem__(self, name: str) -> float:
        try:
            return self._mapping[name]
        except KeyError as err:
            raise ValueError(f"Parameter {name} does not exist") from err

    def update_tree(self) -> None:
        """update names and values of current and parent nodes"""

        def items_walk(
            parameters: _Parameters,
        ) -> Iterator[tuple[tuple[str, float], ...]]:
            yield tuple(parameters._mapping.items())
            for leaf in parameters._leaves.values():
                yield tuple(chain.from_iterable(items_walk(leaf)))

        generator = chain.from_iterable(items_walk(self))
        try:
            _k, _v = zip(*generator, strict=False)
            self._all_names = _k
            self._all_values = _v
            # self._allnames, self._allvalues = zip(*generator)
        except StopIteration:
            pass
        # recursively update parent
        if self.parent is not None:
            self.parent.update_tree()


class ParametricModel:
    """
    Base class of every parametric models in ReLife.
    """

    _params: _Parameters
    _baseline_models: dict[str, ParametricModel]

    def __init__(self, **kwparams: float | None) -> None:
        self._params = _Parameters(**kwparams)
        self._baseline_models = {}

    @property
    def params(self) -> Array1D[np.float64]:
        """
        Get the parameters of this model.

        Returns
        -------
        out : 1darray of number
            Model parameters.

        Notes
        -----
        If parameter values are not set, they default to `np.nan` values.
        """
        # np.number includes complex types
        return np.array(self._params.all_values)

    def set_params(self, new_params: ToFloat1D) -> None:
        """
        Set the parameters of this model.

        Parameters
        ----------
        new_params : array-like of floats
            Model parameters.


        Notes
        -----
        `set_params` definition expects an array-like of floats. At runtime,
        complex parameters might be setted temporarily to approximate fitted
        parameters covariance. This is contradictory to the given typing. At
        the moment, we don't see a better solution and we believe that this is
        actually a limitation of what be expressed in the static typesystem.
        """
        # not @params.setter to allow a different type for the values to set
        new_params = np.asarray(new_params)
        assert new_params.ndim == 1
        self._params.set_all_values(tuple(v.item() for v in new_params))

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
        return self._params.all_names

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

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("_baseline_models"):
            return super().__getattribute__("_baseline_models").get(name)
        raise AttributeError(f"{type(self).__name__} has no attribute named {name}")

    @override
    def __setattr__(self, name: str, value: Any):
        # automatically add params of new baseline model
        if isinstance(value, ParametricModel):
            self._baseline_models[name] = value
            self._params.set_leaf(f"{name}.params", value._params)
        super().__setattr__(name, value)


_ParametricModel_T = TypeVar("_ParametricModel_T", bound=ParametricModel)

Ts = TypeVarTuple("Ts")


class FrozenParametricModel(ParametricModel, Generic[_ParametricModel_T, *Ts]):
    """
    Class of every frozen parametric models.

    Frozen models encapsulate additional arguments values allowing to request
    the object without giving them.
    """

    _args: tuple[*Ts]
    _unfrozen_model: _ParametricModel_T

    def __init__(self, model: _ParametricModel_T, *args: *Ts):
        super().__init__()
        if np.any(np.isnan(model.params)):
            raise ValueError(
                f"""
                Can't freeze a model with np.nan parameters. Model params is
                {model.set_params}
                """
            )
        self._unfrozen_model = model
        self._args = args

    @property
    def args(self) -> tuple[*Ts]:
        return self._args

    @args.setter
    def args(self, value: tuple[*Ts]) -> None:
        if len(value) != len(self._args):
            raise ValueError
        self._args = value

    def unfreeze(self) -> _ParametricModel_T:
        return self._unfrozen_model

    @override
    def __getattr__(self, key: str) -> Any:
        frozen_type = self._unfrozen_model.__class__.__name__
        if key == "fit":
            raise AttributeError("Frozen model can't be fit")
        try:
            attr = getattr(self._unfrozen_model, key)
        except AttributeError as err:
            raise AttributeError(
                f"Frozen {frozen_type} has no attribute {key}"
            ) from err

        def wrapper(*args: Any, **kwargs: Any):
            return attr(*(*args, *self.args), **kwargs)

        if inspect.ismethod(attr):
            return wrapper
        return attr


@overload
def is_frozen(model: FrozenParametricModel[ParametricModel, *Ts]) -> Literal[True]: ...
@overload
def is_frozen(
    model: ParametricModel | FrozenParametricModel[ParametricModel, *Ts],
) -> bool: ...
def is_frozen(
    model: ParametricModel | FrozenParametricModel[ParametricModel, *Ts],
) -> bool:
    """
    Checks if model is frozen
    """
    from relife.base import FrozenParametricModel

    return isinstance(model, FrozenParametricModel)


@dataclass
class FittingResults:
    """Fitting results of the parametric_model core."""

    nb_observations: int  #: Number of observations (samples)
    optimal_params: Array1D[np.float64]  #: Optimal parameters values
    success: bool  #: Whether or not the optimizer exited successfully.
    neg_log_likelihood: float = field(
        repr=False
    )  #: Negative log likelihood value at optimal parameters values

    covariance_matrix: Array2D[np.float64] | None = field(
        repr=False, default=None
    )  #: Covariance matrix (computed as the inverse of the Hessian matrix).

    nb_params: int = field(init=False, repr=False)  #: Number of parameters.
    aic: float = field(init=False)  #: Akaike Information Criterion.
    aicc: float = field(
        init=False
    )  #: Akaike Information Criterion with a correction for small sample sizes.
    bic: float = field(init=False)  #: Bayesian Information Criterion.
    se: Array1D[np.float64] | None = field(
        init=False, repr=False
    )  #: Standard error, square root of the diagonal of the covariance matrix
    ic: Array[tuple[int, Literal[2]], np.float64] | None = field(
        init=False, repr=False
    )  #: 95% IC

    def __post_init__(self):
        self.nb_params = self.optimal_params.size
        self.aic = 2 * self.nb_params + 2 * self.neg_log_likelihood
        self.aicc = self.aic + 2 * self.nb_params * (self.nb_params + 1) / (
            self.nb_observations - self.nb_params - 1
        )
        self.bic = (
            np.log(self.nb_observations) * self.nb_params + 2 * self.neg_log_likelihood
        )
        self.se = None
        self.ic = None
        if self.covariance_matrix is not None:
            self.se = np.sqrt(np.diag(self.covariance_matrix))
            self.ic = self.optimal_params.reshape(-1, 1) + stats.norm.ppf(
                (0.05, 0.95)
            ) * self.se.reshape(-1, 1) / np.sqrt(self.nb_observations)  # (p, 2)

    @override
    def __str__(self) -> str:
        fields = {
            "fitted params": self.optimal_params,
            "AIC": self.aic,
            "AICc": self.aicc,
            "BIC": self.bic,
        }
        # Find the maximum field name length for alignment
        max_name_length = max(len(name) for name, _ in fields.items())
        lines: list[str] = []
        for name, value in fields.items():
            # Format arrays to be more compact
            if isinstance(value, np.ndarray):
                value_str = f"[{', '.join(f'{x:.6g}' for x in value)}]"
            else:
                value_str = f"{value:.6g}" if isinstance(value, float) else str(value)
            lines.append(f"{name:<{max_name_length}} : {value_str}")
        return "\n".join(lines)


M = TypeVar("M", bound=ParametricModel)
D = TypeVar("D")


@dataclass
class OptimizerConfig:
    x0: ToFloat | ToFloat1D
    scipy_minimize_options: dict[str, Any] = field(default_factory=dict)
    covariance_method: Literal["cs", "2point", "exact", False] = False


class MaximumLikelihoodOptimizer(Generic[M, D], ABC):
    """
    Abstract maximum likelihood optimizer.

    Notes
    -----
    Jacobian and hessian are not required but they can be implemented in
    concrete likelihoods. To use the jacobian or hessian implementations in the
    likelihood, pass them into `config["scipy_minimize_options"]`.

    Attributes
    ----------
    nb_observations : int
        The number of observations.
    """

    model: M
    data: D
    config: OptimizerConfig

    @property
    @abstractmethod
    def nb_observations(self) -> int: ...

    @abstractmethod
    def negative_log(self, params: Array1D[np.float64]) -> float:
        """
        Negative log likelihood.

        Parameters
        ----------
        params : 1d array
            Parameters values.

        Returns
        -------
        out : np.float64
            Negative log likelihood value.
        """

    def optimize(self) -> FittingResults:
        """
        Search parameters values that maximize the likelihood given data.

        Returns
        -------
        out : FittingResults
            An object that encapsulates optimal parameters and fitting
            information (AIC, variance, etc.).
        """

        optimizer = minimize(
            self.negative_log,
            self.config.x0,
            **self.config.scipy_minimize_options,
        )

        fitting_results = FittingResults(
            self.nb_observations,
            np.copy(optimizer.x),
            optimizer.success,
            optimizer.fun,
        )

        if not fitting_results.success:
            warnings.warn(
                "The negative log-likelihood minimization did not exited successfully.",
                stacklevel=2,
            )

        if self.config.covariance_method is False:
            return fitting_results

        jac = self.config.scipy_minimize_options.get("jac", None)
        hess = self.config.scipy_minimize_options.get("hess", None)
        if jac is not None and self.config.covariance_method != "exact":
            fitting_results.covariance_matrix = approx_parameters_covariance(
                fitting_results.optimal_params,
                jac,
                method=self.config.covariance_method,
            )
        if hess is not None and self.config.covariance_method == "exact":
            fitting_results.covariance_matrix = np.linalg.pinv(
                hess(fitting_results.optimal_params)
            )
        return fitting_results


def approx_parameters_covariance(
    params: Array1D[np.float64],
    jac_negative_log: Callable[[Array1D[np.number]], Array1D[np.number]],
    method: Literal["2point", "cs"] = "cs",
) -> Array2D[np.float64] | None:
    """
    Approximate parameters covariance.

    Parameters
    ----------
    params : 1darray of float
        The parameters values.
    jac_negative_log : callable
        A function taking 1d array of numbers and returning 1d array of numbers.
    method : "2point" or "cs", default to "cs"
        The approximation method to use.
    """

    size = params.size
    eps = 1e-6
    hess = np.empty((size, size), dtype=np.float64)

    # hessian 2 point
    if method == "2point":
        for i in range(size):
            hess[i] = approx_fprime(
                params,
                lambda x: jac_negative_log(x)[i],
                eps,
            )
        return hess
    # hessian cs
    u = eps * 1j * np.eye(size)
    complex_params = params.astype(np.complex64)  # change params to complex
    for i in range(size):
        for j in range(i, size):
            hess[i, j] = np.imag(jac_negative_log(complex_params + u[i])[j]) / eps
            if i != j:
                hess[j, i] = hess[i, j]
    covariance_matrix = None
    try:
        covariance_matrix = np.linalg.pinv(hess).astype(np.float64)
    except Exception as err:
        warnings.warn(
            f"""
            Failed to compute parameters covariance due to non-invertible
            hessian matrix. Numpy pseudo-inversion algorithm returned : {err}

            You can skip parameters covariance computation by setting
            covariance_method to False. 
            """,
            stacklevel=2,
        )

    return covariance_matrix
