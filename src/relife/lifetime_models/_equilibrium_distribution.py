from typing import final

import numpy as np
from optype.numpy import ArrayND
from scipy.optimize import newton
from typing_extensions import override

from relife.quadratures import legendre_quadrature
from relife.typing import VT

from ._base import ParametricLifetimeModel


@final
class EquilibriumDistribution(ParametricLifetimeModel[*tuple[VT, ...]]):
    r"""
    Equilibrium distribution.

    The equilibirum distribution is the distribution that makes the renewal process
    stationnary.

    Parameters
    ----------
    baseline : any parametric lifetime model
        Lifetime model.

    References
    ----------
    .. [1] Ross, S. M. (1996). Stochastic stochastic_process. New York: Wiley.
    """

    baseline: ParametricLifetimeModel[*tuple[VT, ...]]

    def __init__(
        self,
        baseline: ParametricLifetimeModel[*tuple[VT, ...]],
    ):
        super().__init__()
        self.baseline = baseline

    @override
    def cdf(
        self,
        time: VT,
        *args: VT,
    ) -> np.float64 | ArrayND[np.float64]:
        return legendre_quadrature(
            lambda x: np.asarray(self.baseline.sf(x, *args), dtype=float), 0, time
        ) / self.baseline.mean(*args)

    @override
    def sf(
        self,
        time: VT,
        *args: VT,
    ) -> np.float64 | ArrayND[np.float64]:
        return 1 - self.cdf(time, *args)

    @override
    def pdf(
        self,
        time: VT,
        *args: VT,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.baseline.sf(time, *args) / self.baseline.mean(*args)

    @override
    def hf(
        self,
        time: VT,
        *args: VT,
    ) -> np.float64 | ArrayND[np.float64]:
        return 1 / self.baseline.mrl(time, *args)

    @override
    def chf(
        self,
        time: VT,
        *args: VT,
    ) -> np.float64 | ArrayND[np.float64]:
        return -np.log(self.sf(time, *args))

    @override
    def isf(
        self,
        probability: VT,
        *args: VT,
    ) -> np.float64 | ArrayND[np.float64]:
        def func(x: ArrayND[np.float64]) -> np.float64:
            return np.sum(self.sf(x, *args) - probability)

        return newton(
            func,
            x0=np.asarray(self.baseline.isf(probability, *args)),
            args=args,
        )

    @override
    def ichf(
        self,
        cumulative_hazard_rate: VT,
        *args: VT,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.isf(np.exp(-cumulative_hazard_rate), *args)
