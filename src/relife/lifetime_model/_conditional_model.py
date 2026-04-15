from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, TypeVarTuple

import numpy as np
import numpydoc.docscrape as docscrape  # pyright: ignore[reportMissingTypeStubs]
from optype.numpy import Array, ArrayND, AtMost2D
from typing_extensions import override

from relife.utils import to_column_2d

from ._base import (
    FrozenParametricLifetimeModel,
    ParametricLifetimeModel,
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


class AgeReplacementModel(
    ParametricLifetimeModel[*tuple[ST | NumpyST | ArrayND[NumpyST], *Ts]]
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

    baseline: ParametricLifetimeModel[*Ts]

    def __init__(self, baseline: ParametricLifetimeModel[*Ts]):
        super().__init__()
        self.baseline = baseline

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def sf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        return np.where(time < ar, self.baseline.sf(time, *args), 0.0)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        return np.where(time < ar, self.baseline.hf(time, *args), 0.0)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def cdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        return super().cdf(time, *(ar, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        return np.where(time < ar, self.baseline.chf(time, *args), 0.0)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def isf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        return np.minimum(self.baseline.isf(probability, *args), ar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def ichf(
        self,
        cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        return np.minimum(self.baseline.ichf(cumulative_hazard_rate, *args), ar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def pdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def mrl(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        ub = np.array(np.inf)
        # ar.shape == (m, 1)
        mask = time >= ar  # (m, 1) or (m, n)
        if np.any(mask):
            time, ub = np.broadcast_arrays(time, ub)
            time = np.ma.MaskedArray(time, mask)  # (m, 1) or (m, n)
            ub = np.ma.MaskedArray(ub, mask)  # (m, 1) or (m, n)
        mu = self.ls_integrate(
            lambda x: np.asarray(x - time, dtype=float), time, ub, ar, *args, deg=10
        ) / self.sf(time, ar, *args)  # () or (n,) or (m, n)
        mu = np.ma.filled(mu, 0)
        return np.ma.getdata(mu)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def ppf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        return self.isf(1 - probability, ar, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def median(
        self, ar: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        return self.ppf(np.array(0.5), ar, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def rvs(
        self,
        size: int | tuple[int, int],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> np.float64 | ArrayND[np.float64]:
        args_add_ar = (ar, *args)
        return super().rvs(
            size,
            *args_add_ar,
            seed=seed,
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def ls_integrate(
        self,
        func: Callable[
            [ST | NumpyST | ArrayND[NumpyST]],
            np.float64 | ArrayND[np.float64],
        ],
        a: ST | NumpyST | ArrayND[NumpyST],
        b: ST | NumpyST | ArrayND[NumpyST],
        ar: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        b = np.minimum(ar, b)
        integration = self.baseline.ls_integrate(func, a, b, *args, deg=deg)
        if func(ar).ndim == 2 and integration.ndim == 1:
            integration = integration.reshape(-1, 1)
        return integration + np.where(
            b == ar, func(ar) * self.baseline.sf(ar, *args), 0
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def moment(
        self, n: int, ar: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        return self.ls_integrate(
            lambda x: np.asarray(x**n, dtype=float),
            np.float64(0),
            np.inf,
            ar,
            *args,
            deg=100,
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def mean(
        self, ar: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        return self.moment(1, ar, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def var(
        self, ar: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        ar = to_column_2d(ar)
        return self.moment(2, ar, *args) - self.moment(1, ar, *args) ** 2

    def freeze(
        self, ar: ST | NumpyST | Array[AtMost2D, NumpyST], *args: *Ts
    ) -> FrozenParametricLifetimeModel[
        *tuple[ST | NumpyST | Array[AtMost2D, NumpyST], *Ts]
    ]:
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
    ParametricLifetimeModel[*tuple[ST | NumpyST | ArrayND[NumpyST], *Ts]]
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

    baseline: ParametricLifetimeModel[*Ts]

    def __init__(self, baseline: ParametricLifetimeModel[*Ts]):
        super().__init__()
        self.baseline = baseline

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def sf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        return super().sf(time, a0, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def pdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        return super().pdf(time, a0, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def isf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        a0 = to_column_2d(a0)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def chf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def cdf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        return super().cdf(time, *(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def hf(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        return self.baseline.hf(a0 + time, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def ichf(
        self,
        cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
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
        size: int | tuple[int, int],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        args_add_a0 = (a0, *args)
        return super().rvs(
            size,
            *args_add_a0,
            seed=seed,
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def ls_integrate(
        self,
        func: Callable[
            [ST | NumpyST | ArrayND[NumpyST]],
            np.float64 | ArrayND[np.float64],
        ],
        a: ST | NumpyST | ArrayND[NumpyST],
        b: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        return super().ls_integrate(func, a, b, *(a0, *args), deg=deg)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def mean(
        self, a0: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        return super().mean(*(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def median(
        self, a0: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        return super().median(*(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def var(
        self, a0: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        return super().var(*(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def moment(
        self, n: int, a0: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        return super().moment(n, *(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def mrl(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        return super().mrl(time, *(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def ppf(
        self,
        probability: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        a0 = to_column_2d(a0)
        return super().ppf(probability, *(a0, *args))

    def freeze(
        self, a0: ST | NumpyST | Array[AtMost2D, NumpyST], *args: *Ts
    ) -> FrozenParametricLifetimeModel[
        *tuple[ST | NumpyST | Array[AtMost2D, NumpyST], *Ts]
    ]:
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
