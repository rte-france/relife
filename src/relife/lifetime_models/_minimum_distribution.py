from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Literal, Self, final

import numpy as np
from optype.numpy import Array, Array1D, ArrayND
from typing_extensions import override

from relife.typing import VT

from ._base import (
    FittableParametricLifetimeModel,
    LifetimeLikelihood,
)


@final
class MinimumDistribution(FittableParametricLifetimeModel[*tuple[VT, ...]]):
    r"""
    Series structure of n identical and independent components.

    The hazard function of the system is given by:

    .. math::

        h(t) = n \cdot  h_0(t)

    where :math:`h_0` is the baseline hazard function of the components.

    Parameters
    ----------
    baseline : lifetime distribution or regression
        Lifetime model.

    Examples
    --------

    Computing the survival (or reliability) function for 3 structures of 3,6 and
    9 identical and idependent components:

    .. code-block::

        model = MinimumDistribution(Weibull(2, 0.05))
        t = np.arange(0, 10, 0.1)
        n = np.array([3, 6, 9]).reshape(-1, 1)
        model.sf(t, n)
    """

    baseline: FittableParametricLifetimeModel[*tuple[VT, ...]]
    n: int

    def __init__(
        self,
        baseline: FittableParametricLifetimeModel[*tuple[VT, ...]],
        n: int,
    ):
        super().__init__()
        self.n = n
        self.baseline = baseline

    @override
    def sf(
        self,
        time: VT,
        *args: VT,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().sf(time, *args)

    @override
    def pdf(
        self,
        time: VT,
        *args: VT,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().pdf(time, *args)

    @override
    def hf(
        self,
        time: VT,
        *args: VT,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.n * self.baseline.hf(time, *args)

    @override
    def chf(
        self,
        time: VT,
        *args: VT,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.n * self.baseline.chf(time, *args)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: VT,
        *args: VT,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.baseline.ichf(cumulative_hazard_rate / self.n, *args)

    @override
    def ls_integrate(
        self,
        func: Callable[
            Concatenate[VT, ...],
            np.float64 | ArrayND[np.float64],
        ],
        a: VT,
        b: VT,
        *args: VT,
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().ls_integrate(func, a, b, *args, deg=deg)

    @override
    def dhf(
        self,
        time: VT,
        *args: VT,
    ) -> ArrayND[np.float64]:
        return self.n * self.baseline.dhf(time, *args)

    @override
    def jac_chf(
        self,
        time: VT,
        *args: VT,
    ) -> ArrayND[np.float64]:
        return self.n * self.baseline.jac_chf(time, *args)

    @override
    def jac_hf(
        self,
        time: VT,
        *args: VT,
    ) -> ArrayND[np.float64]:
        return self.n * self.baseline.jac_chf(time, *args)

    @override
    def init_likelihood(
        self,
        time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64],
        args: Sequence[Array1D[np.float64]] | None = None,
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> LifetimeLikelihood:
        likelihood = self.baseline.init_likelihood(time, args, event, entry, **kwargs)
        likelihood.model = MinimumDistribution(likelihood.model, self.n)
        return likelihood

    def fit(
        self,
        time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64],
        args: Sequence[Array1D[np.float64]] | None = None,
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> Self:

        optimizer = self.init_likelihood(time, args, event, entry, **kwargs)
        self.fitting_results = optimizer.optimize()
        self.set_params(self.fitting_results.optimal_params)

        return self
