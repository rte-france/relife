"""Base classes for all parametric models."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import (
    Any,
    Literal,
    Self,
    final,
)

import numpy as np
from optype.numpy import Array, Array1D, Array2D, ToFloat1D
from scipy import stats
from typing_extensions import override

__all__ = ["ParametricModel"]


@final
class _Parameters:
    """
    Dict-like tree structured of parameters.

    Every ``ParametricModel`` are composed of a ``_Parameters`` instance.
    """

    _leaves: dict[str, _Parameters]
    _mapping: dict[str, float]

    def __init__(self, **kwargs: float | None) -> None:
        self._leaves = {}
        self._mapping = {}
        if bool(kwargs):
            self._mapping = {
                k: v if v is not None else np.nan for k, v in kwargs.items()
            }

    def _iter_values(self) -> Iterator[float]:
        yield from self._mapping.values()
        for leaf in self._leaves.values():
            yield from leaf._iter_values()

    def _iter_names(self) -> Iterator[str]:
        yield from self._mapping.keys()
        for leaf in self._leaves.values():
            yield from leaf._iter_names()

    @property
    def all_values(self) -> tuple[float, ...]:
        return tuple(self._iter_values())

    @property
    def all_names(self) -> tuple[str, ...]:
        return tuple(self._iter_names())

    @property
    def size(self) -> int:
        return len(self._mapping) + sum(leaf.size for leaf in self._leaves.values())

    def set_leaf(self, leaf_name: str, leaf: Self) -> None:
        """
        set a leaf or new leaf
        """
        self._leaves[leaf_name] = leaf

    def set_all_values(self, values: tuple[float | None, ...]) -> None:
        """Set values of the whole parameter tree."""
        if len(values) != self.size:
            raise ValueError(f"Expected {self.size} values but got {len(values)}")
        iterator = iter(np.nan if v is None else v for v in values)
        self._set_values_from(iterator)  # consume values and updates _mapping

    def _set_values_from(self, iterator: Iterator[float]) -> None:
        for name in self._mapping:
            self._mapping[name] = next(iterator)
        for leaf in self._leaves.values():
            leaf._set_values_from(iterator)


class ParametricModel:
    """
    Base class of every parametric models in ReLife.


    Examples
    --------
    >>> class ModelA(ParametricModel):
    ...     def __init__(self, a, b):
    ...         super().__init__(a=a, b=b)
    >>> class ModelB(ParametricModel):
    ...     def __init__(self, baseline : ModelA):
    ...         super().__init__()
    ...         self.baseline = baseline
    >>> model_a = ModelA(1, 2)
    >>> model_b = ModelB(model_a)
    >>> model_b.get_params()
    array([1, 2])
    """

    _params: _Parameters
    fitting_results: FittingResults | None

    def __init__(self, **kwparams: float | None) -> None:
        self._params = _Parameters(**kwparams)
        self.fitting_results = None

    def is_parametrized(self) -> bool:
        return bool(~np.all(np.isnan(self.get_params())))

    def is_fitted(self) -> bool:
        return self.fitting_results is not None

    def get_params(self) -> Array1D[np.float64]:
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

    def get_params_names(self) -> tuple[str, ...]:
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

    @override
    def __setattr__(self, name: str, value: Any):
        # automatically add params of new component_model
        if isinstance(value, ParametricModel):
            # a reference of component._params is kept in the _Parameters tree
            # thus changing model params will affect each component params
            self._params.set_leaf(f"{name}.params", value._params)
        super().__setattr__(name, value)


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
