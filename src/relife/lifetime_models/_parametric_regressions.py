"""Lifetime regression

Notes
-----
This module contains two parametric lifetime regressions.
ProportionalHazard is not Cox regression (Cox is semiparametric).
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Sequence
from typing import Any, Literal, Self, TypeAlias, TypeGuard, final

import numpy as np
import numpydoc.docscrape as docscrape  # pyright: ignore[reportMissingTypeStubs]
from optype.numpy import (
    Array,
    Array1D,
    ArrayND,
    AtMost2D,
    is_array_1d,
)
from scipy.optimize import Bounds
from typing_extensions import override

from relife.base import FitConfig, ParametricModel

from ._base import (
    FittableParametricLifetimeModel,
    FrozenParametricLifetimeModel,
    LifetimeData,
    LifetimeLikelihood,
    approx_ls_integrate,
    approx_mean,
    approx_moment,
    approx_var,
    document_args,
)
from ._distributions import (
    Gamma,
    LifetimeDistribution,
    get_distrib_params_bounds,
    init_distrib_params_from_lifetimes,
)

__all__ = [
    "ParametricAcceleratedFailureTime",
    "ParametricProportionalHazard",
]

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


@final
class LinearCovarEffect(ParametricModel):
    """
    Covariates effect.

    Parameters
    ----------
    coefficients : tuple of float, default is (None,)
        Coefficients of the covariates effect.
    """

    def __init__(self, coefficients: tuple[ST | None, ...] = (None,)):
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


def _is_array1d_sequence(
    val: Sequence[object],
) -> TypeGuard[Sequence[Array1D[np.float64]]]:
    return all(is_array_1d(x) for x in val)


class ParametricLifetimeRegression(
    FittableParametricLifetimeModel[*tuple[ST | NumpyST | ArrayND[NumpyST], ...]], ABC
):
    """
    Base class for lifetime regression.
    """

    baseline: LifetimeDistribution
    covar_effect: LinearCovarEffect

    def __init__(
        self,
        baseline: LifetimeDistribution,
        coefficients: tuple[ST | None, ...] = (None,),
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
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def sf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().sf(time, *covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
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
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def cdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().cdf(time, *covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def pdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().pdf(time, *covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def ppf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().ppf(probability, *covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def median(
        self, *covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().median(*covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def jac_sf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        return -self.jac_chf(time, *covar) * self.sf(time, *covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def jac_cdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        return super().jac_cdf(time, *covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
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
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def rvs(
        self,
        size: int | tuple[int, ...],
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

    def ls_integrate(
        self,
        func: Callable[
            [ST | NumpyST | ArrayND[NumpyST]],
            np.float64 | ArrayND[np.float64],
        ],
        a: ST | NumpyST | ArrayND[NumpyST],
        b: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_ls_integrate(self, func, a, b, args=covar, deg=deg)

    def moment(
        self, n: int, *covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_moment(self, n, args=covar)

    def mean(
        self, *covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_mean(self, args=covar)

    def var(
        self, *covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_var(self, args=covar)

    def freeze(
        self, *covar: ST | NumpyST | Array[AtMost2D, NumpyST]
    ) -> FrozenParametricLifetimeModel:
        """
        Freeze regression covar.

        Parameters
        ----------
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape `(nb_coef,)` or `(m, nb_coef)`.

        Returns
        -------
        out: frozen regression
            The same object but with `covar` stored as object data. Calling methods
            from the frozen regression does not need `covar`.
        """  # noqa: E501
        return FrozenParametricLifetimeModel(self, *covar)

    def _get_covar_fit(**kwargs: Any) -> Sequence[Array1D[np.float64]]:
        if "covar" not in kwargs:
            raise ValueError("Expected covar.")
        covar_sequence = kwargs["covar"]
        # typeguards
        assert _is_array1d_sequence(covar_sequence) or is_array_1d(covar_sequence)
        if not isinstance(covar_sequence, Sequence):
            covar_sequence = (covar_sequence,)
        return covar_sequence

    @override
    def init_likelihood(
        self,
        time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> LifetimeLikelihood[Self]:
        covar_sequence = self._get_covar_fit(**kwargs)
        regression = type(self)(
            type(self.baseline)(), coefficients=(0.0,) * len(covar_sequence)
        )  # init new regression object with appropriate number of covar
        lifetime_data = LifetimeData(time, event, entry, covar_sequence)
        x0 = kwargs.get(
            "x0", init_regression_params_from_lifetimes(regression, lifetime_data)
        )
        regression.set_params(x0)
        config = FitConfig(x0)
        config.scipy_minimize_options["bounds"] = kwargs.get(
            "bounds", get_regression_params_bounds(regression)
        )
        config.scipy_minimize_options["method"] = kwargs.get("method", "L-BFGS-B")
        config.covariance_method = kwargs.get(
            "covariance_method",
            "2point" if isinstance(regression.baseline, Gamma) else "cs",
        )
        optimizer = LifetimeLikelihood(regression, lifetime_data, config)
        return optimizer

    @override
    def fit(
        self,
        time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> Self:
        covar_sequence = self._get_covar_fit(**kwargs)
        self.covar_effect = LinearCovarEffect((None,) * len(covar_sequence))
        return super().fit(time, event, entry, **kwargs)


def init_regression_params_from_lifetimes(
    model: ParametricLifetimeRegression, data: LifetimeData
) -> Array1D[np.float64]:
    param0 = np.zeros_like(model.get_params(), dtype=np.float64)
    param0[-model.baseline.get_params().size :] = init_distrib_params_from_lifetimes(
        model.baseline, data
    )
    return param0


def get_regression_params_bounds(model: ParametricLifetimeRegression) -> Bounds:
    nb_coefficients = model.covar_effect.get_params().size
    lb = np.concatenate(
        (
            np.full(nb_coefficients, -np.inf),
            get_distrib_params_bounds(
                model.baseline
            ).lb,  # baseline has _params_bounds according to typing
        )
    )
    ub = np.concatenate(
        (
            np.full(nb_coefficients, np.inf),
            get_distrib_params_bounds(model.baseline).ub,
        )
    )
    return Bounds(lb, ub)


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
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def jac_hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        g = self.covar_effect.g(*covar)
        jac_g = self.covar_effect.jac_g(*covar)  # (nb_coef, ...)
        baseline_hf = self.baseline.hf(time)
        baseline_jac_hf = self.baseline.jac_hf(time)  # (p, ...)
        return np.concatenate(
            (
                baseline_hf * jac_g,  #  (nb_coef, ...)
                g * baseline_jac_hf,  # (p, ...)
            ),
            axis=0,
        )  # (p + nb_coef, ...)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def jac_chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        g = self.covar_effect.g(*covar)
        jac_g = self.covar_effect.jac_g(*covar)  # (nb_coef, ...)
        baseline_chf = self.baseline.chf(time)
        baseline_jac_chf = np.asarray(self.baseline.jac_chf(time))  # (p, ...)
        return np.concatenate(
            (
                baseline_chf * jac_g,  #  (nb_coef, ...)
                g * baseline_jac_chf,  # (p, ...)
            ),
            axis=0,
        )  # (p + nb_coef, ...)


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
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def dhf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        t0 = time / self.covar_effect.g(*covar)
        return self.baseline.dhf(t0) / self.covar_effect.g(*covar) ** 2

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def jac_hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        g = self.covar_effect.g(*covar)
        jac_g = self.covar_effect.jac_g(*covar)  # (nb_coef, ...)
        t0 = time / g
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
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def jac_chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        g = self.covar_effect.g(*covar)
        jac_g = self.covar_effect.jac_g(*covar)  # (nb_coef, ...)
        t0 = time / g
        baseline_jac_chf_t0 = self.baseline.jac_chf(t0)  # (p, ...)
        baseline_hf_t0 = self.baseline.hf(t0)
        return np.concatenate(
            (
                -jac_g / g * t0 * baseline_hf_t0,  #  (nb_coef, ...)
                baseline_jac_chf_t0,  # (p, ...)
            ),
            axis=0,
        )  # (p + nb_coef, ...)
