from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import TypeVarTuple

import numpy as np
import numpydoc.docscrape as docscrape  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from typing_extensions import override

from relife.typing import (
    AnyFloat,
    AnyParametricLifetimeModel,
    NumpyFloat,
    Seed,
)
from relife.utils import reshape_1d_arg

from ._base import FrozenParametricLifetimeModel, ParametricLifetimeModel, document_args

__all__: list[str] = ["AgeReplacementModel", "LeftTruncatedModel"]

Ts = TypeVarTuple("Ts")

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


class AgeReplacementModel(ParametricLifetimeModel[*tuple[AnyFloat, *Ts]]):
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

    baseline: AnyParametricLifetimeModel[*Ts]

    def __init__(self, baseline: AnyParametricLifetimeModel[*Ts]):
        super().__init__()
        self.baseline = baseline

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def sf(self, time: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        return np.where(time < ar, self.baseline.sf(time, *args), 0.0)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def hf(self, time: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        return np.where(time < ar, self.baseline.hf(time, *args), 0.0)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def cdf(self, time: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        return super().cdf(time, *(ar, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def chf(self, time: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        return np.where(time < ar, self.baseline.chf(time, *args), 0.0)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def isf(self, probability: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        return np.minimum(self.baseline.isf(probability, *args), ar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def ichf(
        self, cumulative_hazard_rate: AnyFloat, ar: AnyFloat, *args: *Ts
    ) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        return np.minimum(self.baseline.ichf(cumulative_hazard_rate, *args), ar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def pdf(self, time: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def mrl(self, time: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        ub = np.array(np.inf)
        # ar.shape == (m, 1)
        mask = time >= ar  # (m, 1) or (m, n)
        if np.any(mask):
            time, ub = np.broadcast_arrays(time, ub)
            time = np.ma.MaskedArray(time, mask)  # (m, 1) or (m, n)
            ub = np.ma.MaskedArray(ub, mask)  # (m, 1) or (m, n)
        mu = self.ls_integrate(
            lambda x: x - time, time, ub, ar, *args, deg=10
        ) / self.sf(time, ar, *args)  # () or (n,) or (m, n)
        np.ma.filled(mu, 0)
        return np.ma.getdata(mu)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def ppf(self, probability: AnyFloat, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        return self.isf(1 - probability, ar, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def median(self, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        return self.ppf(np.array(0.5), ar, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def rvs(
        self,
        size: int | tuple[int, int],
        ar: AnyFloat,
        *args: *Ts,
        seed: Seed | None = None,
    ) -> NumpyFloat:
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
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: AnyFloat,
        b: AnyFloat,
        ar: AnyFloat,
        *args: *Ts,
        deg: int = 10,
    ) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        b = np.minimum(ar, b)
        integration = self.baseline.ls_integrate(func, a, b, *args, deg=deg)
        if func(ar).ndim == 2 and integration.ndim == 1:
            integration = integration.reshape(-1, 1)
        return integration + np.where(
            b == ar, func(ar) * self.baseline.sf(ar, *args), 0
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def moment(self, n: int, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        return self.ls_integrate(
            lambda x: x**n,
            np.float64(0),
            np.inf,
            ar,
            *args,
            deg=100,
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def mean(self, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        return self.moment(1, ar, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_ar_args_docstring)
    def var(self, ar: AnyFloat, *args: *Ts) -> NumpyFloat:
        ar = reshape_1d_arg(ar)
        return self.moment(2, ar, *args) - self.moment(1, ar, *args) ** 2

    def freeze(
        self, ar: AnyFloat, *args: *Ts
    ) -> FrozenParametricLifetimeModel[*tuple[AnyFloat, *Ts]]:
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


class LeftTruncatedModel(ParametricLifetimeModel[*tuple[AnyFloat, *Ts]]):
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

    baseline: AnyParametricLifetimeModel[*Ts]

    def __init__(self, baseline: AnyParametricLifetimeModel[*Ts]):
        super().__init__()
        self.baseline = baseline

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def sf(self, time: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
        return super().sf(time, a0, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def pdf(self, time: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
        return super().pdf(time, a0, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def isf(self, probability: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        a0 = reshape_1d_arg(a0)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def chf(self, time: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def cdf(self, time: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
        return super().cdf(time, *(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def hf(self, time: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
        return self.baseline.hf(a0 + time, *args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def ichf(
        self, cumulative_hazard_rate: AnyFloat, a0: AnyFloat, *args: *Ts
    ) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
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
        a0: AnyFloat,
        *args: *Ts,
        seed: Seed | None = None,
    ) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
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
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: AnyFloat,
        b: AnyFloat,
        a0: AnyFloat,
        *args: *Ts,
        deg: int = 10,
    ) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
        return super().ls_integrate(func, a, b, *(a0, *args), deg=deg)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def mean(self, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
        return super().mean(*(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def median(self, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
        return super().median(*(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def var(self, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
        return super().var(*(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def moment(self, n: int, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
        return super().moment(n, *(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def mrl(self, time: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
        return super().mrl(time, *(a0, *args))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_a0_args_docstring)
    def ppf(self, probability: AnyFloat, a0: AnyFloat, *args: *Ts) -> NumpyFloat:
        a0 = reshape_1d_arg(a0)
        return super().ppf(probability, *(a0, *args))

    def freeze(
        self, a0: AnyFloat, *args: *Ts
    ) -> FrozenParametricLifetimeModel[*tuple[AnyFloat, *Ts]]:
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


def build_conditional_lifetime_model(lifetime_model: AnyParametricLifetimeModel[*Ts], a0: NumpyFloat | None = None, ar: NumpyFloat | None = None) -> AnyParametricLifetimeModel[*Ts]:
    # TODO: use copy method of parametricmodel
    applied_model = deepcopy(lifetime_model)
    if ar is not None:
        applied_model = AgeReplacementModel(applied_model).freeze(ar)
    if a0 is not None:
        applied_model = LeftTruncatedModel(applied_model).freeze(a0)
    return applied_model
