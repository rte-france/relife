"""Lifetime regression

Notes
-----
This module contains two parametric lifetime regressions.
ProportionalHazard is not Cox regression (Cox is semiparametric).
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import Any, Self, TypeAlias, final

import numpy as np
import numpydoc.docscrape as docscrape  # pyright: ignore[reportMissingTypeStubs]
from optype.numpy import (
    Array,
    Array1D,
    Array2D,
    ArrayND,
    AtMost2D,
    is_array_2d,
)
from scipy.optimize import Bounds
from typing_extensions import override

from relife.base import OptimizerConfig, ParametricModel
from relife.utils import to_column_2d_if_1d

from ._base import (
    FittableParametricLifetimeModel,
    FrozenParametricLifetimeModel,
    LifetimeData,
    LifetimeLikelihood,
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


def _broadcast_time_covar(
    time: ST | NumpyST | ArrayND[NumpyST],
    covar: ST | NumpyST | ArrayND[NumpyST],
) -> tuple[ArrayND[np.float64], Array[AtMost2D, np.float64]]:
    time = np.atleast_2d(np.asarray(time, dtype=np.float64))  #  (m, n)
    covar = np.atleast_2d(np.asarray(covar, dtype=np.float64))  #  (m, nb_coef)
    assert is_array_2d(time)  # typeguards
    assert is_array_2d(covar)  # typeguards
    match (time.shape[0], covar.shape[0]):
        case (1, _):
            time = np.repeat(time, covar.shape[0], axis=0)
        case (_, 1):
            covar = np.repeat(covar, time.shape[0], axis=0)
        case (m1, m2) if m1 != m2:
            raise ValueError(
                f"""
                Incompatible time and covar. time has {m1} nb_assets but
                covar has {m2} nb_assets
                """
            )
        case _:
            pass
    return time, covar


def _broadcast_time_covar_shapes(
    time_shape: tuple[int, ...], covar_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    time_shape : (), (n,) or (m, n)
    covar_shape : (), (nb_coef,) or (m, nb_coef)
    """
    match [time_shape, covar_shape]:
        case [(), ()] | [(), (_,)]:
            return ()
        case [(), (m, _)]:
            return m, 1
        case [(n,), ()] | [(n,), (_,)]:
            return (n,)
        case [(n,), (m, _)] | [(m, n), ()] | [(m, n), (_,)]:
            return m, n
        case [(mt, n), (mc, _)] if mt != mc:
            if mt != 1 and mc != 1:
                raise ValueError(
                    f"""
                    Invalid time and covar : time got {mt} nb assets but covar
                    got {mc} nb assets
                    """
                )
            return max(mt, mc), n
        case [(mt, n), (mc, _)] if mt == mc:
            return mt, n
        case _:
            raise ValueError(
                f"Invalid time or covar shape. Got {time_shape} and {covar_shape}"
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

    def __init__(self, coefficients: tuple[ST | None, ...] = (None,)):
        super().__init__(**{f"coef_{i + 1}": v for i, v in enumerate(coefficients)})

    def g(
        self, covar: ST | NumpyST | ArrayND[NumpyST]
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
            If `covar.shape` is `()`, `out` is `float`.
            If `covar.shape` is `(nb_coef,)`, `out.shape` is `()`.
            If `covar.shape` is `(m, nb_coef)`, `out.shape` is `(m, 1)`.
        """
        arr_covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        if arr_covar.ndim > 2:
            raise ValueError(
                f"""
                Invalid covar shape. Expected (nb_coef,) or (m, nb_coef) but
                got {arr_covar.shape}
                """
            )
        covar_nb_coef = arr_covar.size if arr_covar.ndim <= 1 else arr_covar.shape[-1]
        nb_coef = self.get_params().size
        if covar_nb_coef != nb_coef:
            raise ValueError(
                f"""
                Invalid covar. Number of covar does not match number of
                coefficients. Got {nb_coef} nb_coef but covar shape is
                {arr_covar.shape}
                """
            )
        g = np.exp(
            np.sum(self.get_params() * arr_covar, axis=-1, keepdims=True)
        )  # (m, 1)
        if arr_covar.ndim <= 1:
            return np.float64(g.item())
        return g

    def jac_g(self, covar: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        """
        Returns the jacobian of the covariates effect.

        Parameters
        ----------
        covar : float or np.ndarray
            The covariate values

        Returns
        -------
        out : np.ndarray
            If `covar.shape` is `()` or `(nb_coef,)`, `out.shape` is `(nb_coef,)`.
            If `covar.shape` is (m, nb_coef)`, `out.shape` is `(nb_coef, m, 1)`.
        """
        arr_covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        g = self.g(arr_covar)  # () or (m, 1)
        nb_coef = self.get_params().size
        jac = arr_covar.T.reshape(nb_coef, -1, 1) * g  # (nb_coef, m, 1)
        if arr_covar.ndim <= 1:
            jac = jac.reshape(nb_coef)  # (nb_coef,) or (nb_coef, m, 1)
        return jac  # (nb_coef, m, 1)


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
    FittableParametricLifetimeModel[ST | NumpyST | ArrayND[NumpyST]], ABC
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
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().sf(time, covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def isf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        cumulative_hazard_rate = -np.log(
            np.clip(probability, 0, 1 - np.finfo(float).resolution)
        )
        return self.ichf(cumulative_hazard_rate, covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def cdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().cdf(time, covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def pdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().pdf(time, covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def ppf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().ppf(probability, covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def mrl(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().mrl(time, covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def ls_integrate(
        self,
        func: Callable[
            [ST | NumpyST | ArrayND[NumpyST]],
            np.float64 | ArrayND[np.float64],
        ],
        a: ST | NumpyST | ArrayND[NumpyST],
        b: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
        *,
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().ls_integrate(func, a, b, covar, deg=deg)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def moment(
        self, n: int, covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().moment(n, covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def mean(
        self, covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().mean(covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def var(
        self, covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().var(covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def median(
        self, covar: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().median(covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def jac_sf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        return -self.jac_chf(time, covar) * self.sf(time, covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def jac_cdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        return super().jac_cdf(time, covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def jac_pdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        jac = self.jac_hf(time, covar) * self.sf(time, covar) + self.jac_sf(
            time, covar
        ) * self.hf(time, covar)
        return jac

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def rvs(
        self,
        size: int | tuple[int, int],
        covar: ST | NumpyST | ArrayND[NumpyST],
        *,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().rvs(
            size,
            covar,
            seed=seed,
        )

    def freeze(
        self, covar: ST | NumpyST | Array[AtMost2D, NumpyST]
    ) -> FrozenParametricLifetimeModel[ST | NumpyST | Array[AtMost2D, NumpyST]]:
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
        return FrozenParametricLifetimeModel(self, covar)

    @override
    def init_likelihood(
        self,
        time: Array1D[np.float64],
        args: Array1D[Any]
        | Array2D[Any]
        | tuple[Array1D[Any] | Array2D[Any], ...]
        | None = None,
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> LifetimeLikelihood[Self]:
        if not isinstance(args, np.ndarray):
            raise ValueError("args is expected to be covar only.")
        covar = to_column_2d_if_1d(args)
        regression = type(self)(
            type(self.baseline)(), coefficients=(0.0,) * covar.shape[-1]
        )  # init new regression object with appropriate number of covar
        lifetime_data = LifetimeData(time, args, event, entry)
        x0 = kwargs.get(
            "x0", init_regression_params_from_lifetimes(regression, lifetime_data)
        )
        regression.set_params(x0)
        config = OptimizerConfig(x0)
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
        time: Array1D[np.float64],
        args: Array1D[Any]
        | Array2D[Any]
        | tuple[Array1D[Any] | Array2D[Any], ...]
        | None = None,
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> Self:
        if not isinstance(args, np.ndarray):
            raise ValueError("args is expected to be covar only.")
        self.covar_effect = LinearCovarEffect(
            (None,) * to_column_2d_if_1d(np.asarray(args, dtype=np.float64)).shape[-1]
        )  # changes params structure depending on number of covar
        return super().fit(time, args, event, entry, **kwargs)


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
    coefficients
    nb_params
    params
    params_names
    plot


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
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.hf(time)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.chf(time)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def ichf(
        self,
        cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return self.baseline.ichf(cumulative_hazard_rate / self.covar_effect.g(covar))

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def dhf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.dhf(time)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def jac_hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        time = np.asarray(time)  # (), (n,) or (m, n)
        covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        out_shape = _broadcast_time_covar_shapes(
            time.shape, covar.shape
        )  # (), (n,) or (m, n)
        time, covar = _broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar)  # (nb_coef, m, 1)

        baseline_hf = np.asarray(self.baseline.hf(time))  # (m, n)
        # p == baseline.nb_params
        baseline_jac_hf = self.baseline.jac_hf(time)  # (p, m, n)
        jac_g = np.repeat(
            jac_g, baseline_hf.shape[-1], axis=-1
        )  # (nb_coef, m, n) necessary to concatenate

        jac = np.concatenate(
            (
                baseline_hf * jac_g,  #  (nb_coef, m, n)
                g * baseline_jac_hf,  # (p, m, n)
            ),
            axis=0,
        )  # (p + nb_coef, m, n)

        nb_params = self.get_params().size
        jac = jac.reshape((nb_params,) + out_shape)
        return jac

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def jac_chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        time = np.asarray(time)  # (), (n,) or (m, n)
        covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        out_shape = _broadcast_time_covar_shapes(
            time.shape, covar.shape
        )  # (), (n,) or (m, n)
        time, covar = _broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar)  # (nb_coef, m, 1)
        baseline_chf = self.baseline.chf(time)  # (m, n)
        #  p == baseline.nb_params
        baseline_jac_chf = np.asarray(self.baseline.jac_chf(time))  # (p, m, n)
        jac_g = np.repeat(
            jac_g, baseline_chf.shape[-1], axis=-1
        )  # (nb_coef, m, n) necessary to concatenate

        jac = np.concatenate(
            (
                baseline_chf * jac_g,  #  (nb_coef, m, n)
                g * baseline_jac_chf,  # (p, m, n)
            ),
            axis=0,
        )  # (p + nb_coef, m, n)

        nb_params = self.get_params().size
        jac = jac.reshape((nb_params,) + out_shape)
        return jac


@final
class ParametricAcceleratedFailureTime(ParametricLifetimeRegression):
    # noinspection PyUnresolvedReferences
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
    coefficients
    nb_params
    params
    params_names
    plot


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
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.hf(t0) / self.covar_effect.g(covar)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.chf(t0)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def ichf(
        self,
        cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.ichf(cumulative_hazard_rate)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def dhf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.dhf(t0) / self.covar_effect.g(covar) ** 2

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def jac_hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        time = np.asarray(time)  # (), (n,) or (m, n)
        covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        out_shape = _broadcast_time_covar_shapes(
            time.shape, covar.shape
        )  # (), (n,) or (m, n)
        time, covar = _broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar)  # (nb_coef, m, 1)
        t0 = time / g  # (m, n)
        # p == baseline.nb_params
        baseline_jac_hf_t0 = self.baseline.jac_hf(t0)  # (p, m, n)
        baseline_hf_t0 = self.baseline.hf(t0)  # (m, n)
        baseline_dhf_t0 = self.baseline.dhf(t0)  # (m, n)
        jac_g = np.repeat(jac_g, baseline_hf_t0.shape[-1], axis=-1)  # (nb_coef, m, n)

        jac = np.concatenate(
            (
                -jac_g
                / g**2
                * (
                    baseline_hf_t0 + t0 * baseline_dhf_t0
                ),  # (nb_coef, m, n) necessary to concatenate
                baseline_jac_hf_t0 / g,  # (p, m, n)
            ),
            axis=0,
        )  # (p + nb_coef, m, n)

        nb_params = self.get_params().size
        jac = jac.reshape((nb_params,) + out_shape)
        return jac

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def jac_chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        covar: ST | NumpyST | ArrayND[NumpyST],
    ) -> ArrayND[np.float64]:
        time = np.asarray(time)  # (), (n,) or (m, n)
        covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        out_shape = _broadcast_time_covar_shapes(
            time.shape, covar.shape
        )  # (), (n,) or (m, n)
        time, covar = _broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar)  # (nb_coef, m, 1)
        t0 = time / g  #  (m, n)
        # p == baseline.nb_params
        baseline_jac_chf_t0 = self.baseline.jac_chf(t0)  # (p, m, n)
        baseline_hf_t0 = self.baseline.hf(t0)  #  (m, n)
        jac_g = np.repeat(
            jac_g, baseline_hf_t0.shape[-1], axis=-1
        )  # (nb_coef, m, n) necessary to concatenate

        jac = np.concatenate(
            (
                -jac_g / g * t0 * baseline_hf_t0,  #  (nb_coef, m, n)
                baseline_jac_chf_t0,  # (p, m, n)
            ),
            axis=0,
        )  # (p + nb_coef, m, n)

        nb_params = self.get_params().size
        jac = jac.reshape((nb_params,) + out_shape)
        return jac
