"""Lifetime regression

Notes
-----
This module contains two parametric lifetime regressions.
ProportionalHazard is not Cox regression (Cox is semiparametric).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Literal, Self, final

import numpy as np
import numpydoc.docscrape as docscrape  # pyright: ignore[reportMissingTypeStubs]
from optype.numpy import (
    Array,
    Array1D,
    ArrayND,
)
from typing_extensions import override

from relife.base import FittingResults, ParametricModel
from relife.typing import ST, NumpyST

from ._base import (
    ParametricLifetimeModel,
    document_args,
)
from ._distributions import (
    LifetimeDistribution,
)


@final
class LinearCovarEffect(ParametricModel):
    """
    Covariates effect.

    Parameters
    ----------
    coefficients : tuple of float, default is (None,)
        Coefficients of the covariates effect.
    """

    def __init__(self, coefficients: Sequence[ST | None] = (None,)):
        super().__init__(**{f"coef_{i + 1}": v for i, v in enumerate(coefficients)})

    def g(
        self, *covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        """
        Returns the covariates effect.

        Parameters
        ----------
        covar : float or np.ndarray
            The covariate values

        Returns
        -------
        out : np.float64 or np.ndarray
        """
        nb_coef = self.get_params().size
        if len(covar) != nb_coef:
            raise ValueError(
                f"""
                Invalid number of covar. Got {nb_coef} coefficients but {len(covar)} covariates are given.
                """  # noqa: E501
            )
        broadcasted_covar = np.broadcast_arrays(*covar)
        stack_covar = np.stack(broadcasted_covar, axis=-1)
        return np.exp(np.sum(stack_covar * self.get_params(), axis=-1))

    def jac_g(self, *covar: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        """
        Returns the jacobian of the covariates effect.

        Parameters
        ----------
        covar : float or np.ndarray
            The covariate values

        Returns
        -------
        out : np.ndarray
        """
        g = self.g(*covar)
        broadcasted_covar = np.broadcast_arrays(*covar)
        stack_covar = np.stack(broadcasted_covar, axis=0)
        return stack_covar * g

    @override
    def __repr__(self) -> str:
        return f"LinearCovarEffect({self.get_params()!r})"


_covar_docstring = [
    docscrape.Parameter(
        "covar",
        "float or np.ndarray",
        [
            "Covariates values.",
            "float can only be valid if the regression has one coefficients.",
            "Otherwise it must be a ndarray of shape `(nb_coef,)` or `(m, nb_coef)`.",
        ],
    ),
]


class ParametricLifetimeRegression(
    ParametricLifetimeModel[*tuple[ST | NumpyST | ArrayND[NumpyST], ...]], ABC
):
    """
    Base class for lifetime regression.
    """

    baseline: LifetimeDistribution
    covar_effect: LinearCovarEffect
    fitting_results: FittingResults | None

    def __init__(
        self,
        baseline: LifetimeDistribution,
        coefficients: Sequence[ST | None] = (None,),
    ):
        super().__init__()
        self.covar_effect = LinearCovarEffect(coefficients)
        self.baseline = baseline

    def get_coefficients(self) -> Array1D[np.float64]:
        """
        Returns the coefficients values.

        Returns
        -------
        out : ndarray
        """
        return self.covar_effect.get_params()

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def sf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().sf(time, *covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def isf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        cumulative_hazard_rate = -np.log(
            np.clip(probability, 0, 1 - np.finfo(float).resolution)
        )
        return self.ichf(cumulative_hazard_rate, *covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def cdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().cdf(time, *covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def pdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().pdf(time, *covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def ppf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().ppf(probability, *covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def median(
        self, *covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().median(*covar)

    @abstractmethod
    def jac_hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        """
        The jacobian of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are `()`, `(n,)` or `(m, n)`.
        *args
            Any additonal args.

        Returns
        -------
        out : np.ndarray
        """

    @abstractmethod
    def jac_chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        """
        The jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
        *args
            Any additonal args.

        Returns
        -------
        out : np.ndarray
        """

    @abstractmethod
    def dhf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        """
        The derivate of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
        *args
            Any additonal args.

        Returns
        -------
        out : np.ndarray
        """

    def jac_sf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        return -self.jac_chf(time, *covar) * self.sf(time, *covar)

    def jac_cdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        return -self.jac_sf(time, *covar)

    def jac_pdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        jac = self.jac_hf(time, *covar) * self.sf(time, *covar) + self.jac_sf(
            time, *covar
        ) * self.hf(time, *covar)
        return jac

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *covar: ST | NumpyST | ArrayND[NumpyST],
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().rvs(
            size,
            *covar,
            seed=seed,
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def ls_integrate(
        self,
        func: Callable[
            Concatenate[ST | NumpyST | ArrayND[NumpyST], ...],
            np.float64 | ArrayND[np.float64],
        ],
        a: ST | NumpyST | ArrayND[NumpyST],
        b: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().ls_integrate(func, a, b, *covar, deg=deg)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def moment(
        self, n: int, *covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().moment(n, *covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def mean(
        self, *covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().mean(*covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def var(
        self, *covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().var(*covar)

    def fit(
        self,
        time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64],
        covar: Array1D[np.float64] | Sequence[Array1D[np.float64]],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> Self:
        # local import to avoid circular import
        from relife.likelihoods import LifetimeLikelihood

        optimizer = LifetimeLikelihood.from_regression(
            self, time, covar, event, entry, **kwargs
        )
        self.fitting_results = optimizer.optimize()
        self.set_params(self.fitting_results.optimal_params)

        return self


@final
class ParametricProportionalHazard(ParametricLifetimeRegression):
    r"""
    Proportional Hazard regression.

    The cumulative hazard function :math:`H` is linked to the multiplier
    function :math:`g` by the relation:

    .. math::

        H(t, x) = g(\beta, x) H_0(t) = e^{\beta \cdot x} H_0(t)

    where :math:`x` is a vector of covariates, :math:`\beta` is the coefficient
    vector of the effect of covariates, :math:`H_0` is the baseline cumulative
    hazard function [1]_.

    |

    Parameters
    ----------
    baseline : FittableParametricLifetimeModel
        Any lifetime model that can be fitted.
    coefficients : tuple of floats (values can be None), default is (None,)
        Coefficients values of the covariate effects.

    Attributes
    ----------
    baseline : FittableParametricLifetimeModel
        The regression baseline model (lifetime model).
    covar_effect : _CovarEffect
        The regression covariate effect.
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.).
        If the model is not fitted, the value is None.

    References
    ----------
    .. [1] Sun, J. (2006). The statistical analysis of interval-censored failure
        time data (Vol. 3, No. 1). New York: springer.

    See Also
    --------
    regression.AFT : Accelerated Failure Time regression.

    """

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return self.covar_effect.g(*covar) * self.baseline.hf(time)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return self.covar_effect.g(*covar) * self.baseline.chf(time)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def ichf(
        self,
        cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return self.baseline.ichf(cumulative_hazard_rate / self.covar_effect.g(*covar))

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def dhf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        return self.covar_effect.g(*covar) * self.baseline.dhf(time)

    @override
    def jac_hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        ndtime, *ndcovar = np.broadcast_arrays(time, *covar)
        u = self.baseline.hf(ndtime) * self.covar_effect.jac_g(
            *ndcovar
        )  # (nb_coef, ...)
        v = self.covar_effect.g(*ndcovar) * self.baseline.jac_hf(ndtime)  # (p, ...)
        return np.concatenate((u, v), axis=0)  # (p + nb_coef, ...)

    @override
    def jac_chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        ndtime, *ndcovar = np.broadcast_arrays(time, *covar)
        u = self.baseline.chf(ndtime) * self.covar_effect.jac_g(
            *ndcovar
        )  # (nb_coef, ...)
        v = self.covar_effect.g(*ndcovar) * self.baseline.jac_chf(ndtime)
        return np.concatenate((u, v), axis=0)  # (p + nb_coef, ...)

    @override
    def __repr__(self) -> str:
        return f"ParametricProportionalHazard({repr(self.baseline)}, {self.get_coefficients()!r})"  # noqa: E501


@final
class ParametricAcceleratedFailureTime(ParametricLifetimeRegression):
    r"""
    Accelerated failure time regression.

    The cumulative hazard function :math:`H` is linked to the multiplier
    function :math:`g` by the relation:

    .. math::

        H(t, x) = H_0\left(\dfrac{t}{g(\beta, x)}\right) = H_0(t e^{- \beta
        \cdot x})

    where :math:`x` is a vector of covariates, :math:`\beta` is the coefficient
    vector of the effect of covariates, :math:`H_0` is the baseline cumulative
    hazard function [1]_.

    |

    Parameters
    ----------
    baseline : FittableParametricLifetimeModel
        Any lifetime model that can be fitted.
    coefficients : tuple of floats (values can be None), default is (None,)
        Coefficients values of the covariate effects.

    Attributes
    ----------
    baseline : FittableParametricLifetimeModel
        The regression baseline model (lifetime model).
    covar_effect : _CovarEffect
        The regression covariate effect.
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.).
        If the model is not fitted, the value is None.

    References
    ----------
    .. [1] Kalbfleisch, J. D., & Prentice, R. L. (2011). The statistical
        analysis of failure time data. John Wiley & Sons.

    See Also
    --------
    regression.ProportionalHazard : proportional hazard regression
    """

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        t0 = time / self.covar_effect.g(*covar)
        return self.baseline.hf(t0) / self.covar_effect.g(*covar)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        t0 = time / self.covar_effect.g(*covar)
        return self.baseline.chf(t0)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def ichf(
        self,
        cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return self.covar_effect.g(*covar) * self.baseline.ichf(cumulative_hazard_rate)

    @override
    def dhf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        t0 = time / self.covar_effect.g(*covar)
        return self.baseline.dhf(t0) / self.covar_effect.g(*covar) ** 2

    @override
    def jac_hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        ndtime, *ndcovar = np.broadcast_arrays(time, *covar)
        g = self.covar_effect.g(*ndcovar)
        jac_g = self.covar_effect.jac_g(*ndcovar)  # (nb_coef, ...)
        t0 = ndtime / g
        baseline_jac_hf_t0 = self.baseline.jac_hf(t0)  # (p, ...)
        baseline_hf_t0 = self.baseline.hf(t0)
        baseline_dhf_t0 = self.baseline.dhf(t0)
        return np.concatenate(
            (
                -jac_g
                / g**2
                * (
                    baseline_hf_t0 + t0 * baseline_dhf_t0
                ),  # (nb_coef, ...) necessary to concatenate
                baseline_jac_hf_t0 / g,  # (p, ...)
            ),
            axis=0,
        )  # (p + nb_coef, ...)

    @override
    def jac_chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        ndtime, *ndcovar = np.broadcast_arrays(time, *covar)
        g = self.covar_effect.g(*ndcovar)
        jac_g = self.covar_effect.jac_g(*ndcovar)  # (nb_coef, ...)
        t0 = ndtime / g
        baseline_jac_chf_t0 = self.baseline.jac_chf(t0)  # (p, ...)
        baseline_hf_t0 = self.baseline.hf(t0)
        return np.concatenate(
            (
                -jac_g / g * t0 * baseline_hf_t0,  #  (nb_coef, ...)
                baseline_jac_chf_t0,  # (p, ...)
            ),
            axis=0,
        )  # (p + nb_coef, ...)

    @override
    def __repr__(self) -> str:
        return f"ParametricAcceleratedFailureTime({repr(self.baseline)}, {self.get_coefficients()!r})"  # noqa: E501
