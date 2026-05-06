from __future__ import annotations

from collections.abc import Callable
from typing import Concatenate, TypeAlias, TypeVarTuple

import numpy as np
import numpydoc.docscrape as docscrape  # pyright: ignore[reportMissingTypeStubs]
from optype.numpy import Array, ArrayND, AtMost2D
from typing_extensions import override

from relife.utils import to_column_2d_if_1d

from ._base import (
    FrozenParametricLifetimeModel,
    ParametricLifetimeModel,
    approx_ls_integrate,
    approx_mean,
    approx_moment,
    approx_mrl,
    approx_var,
    document_args,
)

__all__: list[str] = ["AgeReplacementModel", "LeftTruncatedModel"]

Ts = TypeVarTuple("Ts")
ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint

_ar_args_docstring = [
    docscrape.Parameter(
        "ar",
        "float or np.ndarray",
        [
            "Age of replacement values.",
            """
            If ndarray, shape can only be `(m,)` because only one age of
            replacement per asset can be given.
            """,
        ],
    ),
    docscrape.Parameter(
        "*args",
        "",
        [
            "Any other arguments needed by the model.",
        ],
    ),
]


AR: TypeAlias = ST | NumpyST | ArrayND[NumpyST]
A0: TypeAlias = ST | NumpyST | ArrayND[NumpyST]
ArgT: TypeAlias = ST | NumpyST | ArrayND[NumpyST]


class AgeReplacementModel(
    ParametricLifetimeModel[*tuple[AR, *tuple[ArgT, ...]]],
):
    r"""
    Age replacement model.

    Lifetime model where the assets are replaced at age :math:`a_r`. This is
    equivalent to the model of :math:`\min(X,a_r)` where :math:`X` is a
    baseline lifetime and :math:`a_r` is the age of replacement.

    Parameters
    ----------
    baseline : any parametric lifetime model (frozen lifetime model works)
        The base lifetime model without conditional probabilities

    Attributes
    ----------
    baseline
    nb_params
    params
    params_names
    plot
    """

    baseline: ParametricLifetimeModel[*tuple[ArgT, ...]]

    def __init__(self, baseline: ParametricLifetimeModel[*tuple[ArgT, ...]]):
        super().__init__()
        self.baseline = baseline

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def sf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return np.where(time < ar, self.baseline.sf(time, *args), 0.0)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return np.where(time < ar, self.baseline.hf(time, *args), 0.0)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def cdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().cdf(time, *(ar, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return np.where(time < ar, self.baseline.chf(time, *args), 0.0)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def isf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return np.minimum(self.baseline.isf(probability, *args), ar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def ichf(
        self,
        cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return np.minimum(self.baseline.ichf(cumulative_hazard_rate, *args), ar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def pdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def ppf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return self.isf(1 - probability, ar, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def median(
        self,
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return self.ppf(0.5, ar, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().rvs(
            size,
            *(ar, *args),
            seed=seed,
        )

    def ls_integrate(
        self,
        func: Callable[
            Concatenate[ST | NumpyST | ArrayND[NumpyST], ...],
            np.float64 | ArrayND[np.float64],
        ],
        a: ST | NumpyST | ArrayND[NumpyST],
        b: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
        b = np.minimum(ar, b)
        integration = approx_ls_integrate(
            self.baseline, func, a, b, args=(ar, *args), deg=deg
        )
        return integration + np.where(
            b == ar, func(ar, *args) * self.baseline.sf(ar, *args), 0
        )

    def moment(
        self,
        n: int,
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_moment(self, n, args=(ar, *args))

    def mean(
        self,
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_mean(self, args=(ar, *args))

    def var(
        self,
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_var(self, args=(ar, *args))

    def mrl(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_mrl(self, time, args=(ar, *args))

    def freeze(
        self,
        ar: ST | NumpyST | Array[AtMost2D, NumpyST],
        *args: ST | NumpyST | Array[AtMost2D, NumpyST],
    ) -> FrozenParametricLifetimeModel:
        """
        Freeze age replacement values and other arguments into the object data.

        Parameters
        ----------
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenParametricModel
        """
        return FrozenParametricLifetimeModel(self, ar, *args)


_a0_args_docstring = [
    docscrape.Parameter(
        "a0",
        "float or np.ndarray",
        [
            "Current ages.",
            "If ndarray, shape can only be `(m,)`.",
        ],
    ),
    docscrape.Parameter(
        "*args",
        "",
        [
            "Any other arguments needed by the model.",
        ],
    ),
]


class LeftTruncatedModel(
    ParametricLifetimeModel[*tuple[A0, *tuple[ArgT, ...]]],
):
    r"""Left truncated model.

    Lifetime model where the assets have already reached the age :math:`a_0`.

    Parameters
    ----------
    baseline : any parametric lifetime model (frozen lifetime model works)
        The base lifetime model without conditional probabilities
    nb_params
    params
    params_names
    plot

    Attributes
    ----------
    baseline
    nb_params
    params
    params_names
    plot
    """

    baseline: ParametricLifetimeModel[*tuple[ArgT, ...]]

    def __init__(self, baseline: ParametricLifetimeModel[*tuple[ArgT, ...]]):
        super().__init__()
        self.baseline = baseline

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def sf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().sf(time, a0, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def pdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().pdf(time, a0, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def isf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        return self.ichf(cumulative_hazard_rate, a0, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def cdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().cdf(time, *(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return self.baseline.hf(a0 + time, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def ichf(
        self,
        cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return (
            self.baseline.ichf(
                cumulative_hazard_rate + self.baseline.chf(a0, *args), *args
            )
            - a0
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def rvs(
        self,
        size: int | tuple[int, ...],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().rvs(
            size,
            *(a0, *args),
            seed=seed,
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def median(
        self,
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d_if_1d(a0)
        return super().median(*(a0, *args))

    def ls_integrate(
        self,
        func: Callable[
            Concatenate[ST | NumpyST | ArrayND[NumpyST], ...],
            np.float64 | ArrayND[np.float64],
        ],
        a: ST | NumpyST | ArrayND[NumpyST],
        b: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_ls_integrate(self.baseline, func, a, b, args=(a0, *args), deg=deg)

    def moment(
        self,
        n: int,
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_moment(self, n, args=(a0, *args))

    def mean(
        self,
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_mean(self, args=(a0, *args))

    def var(
        self,
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_var(self, args=(a0, *args))

    def mrl(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return approx_mrl(self, time, args=(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def ppf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        return super().ppf(probability, *(a0, *args))

    def freeze(
        self,
        a0: ST | NumpyST | Array[AtMost2D, NumpyST],
        *args: ST | NumpyST | Array[AtMost2D, NumpyST],
    ) -> FrozenParametricLifetimeModel:
        """
        Freeze conditional age values and other arguments into the object data.

        Parameters
        ----------
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If
            ndarray, shape can only be (m,) as only one age per asset can be
            given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenLeftTruncatedModel
        """
        return FrozenParametricLifetimeModel(self, a0, *args)
