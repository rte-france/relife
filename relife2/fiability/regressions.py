"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds

from relife2.data import LifetimeSample
from relife2.model import ParametricComponent, ParametricLifetimeModel

Ts = TypeVarTuple("Ts")


class CovarEffect(ParametricComponent):
    def __init__(self, coef: tuple[float, ...] | tuple[None] = (None,)):
        super().__init__()
        self.new_params(**{f"coef_{i}": v for i, v in enumerate(coef)})

    def g(self, covar: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        BLABLABLABLA
        Returns:
            NDArray[np.float64]: BLABLABLABLA
        """
        if covar.shape[-1] != self.nb_params:
            raise ValueError(
                f"Invalid number of covar : expected {self.nb_params}, got {covar.shape[-1]}"
            )
        return np.exp(np.sum(self.params * covar, axis=1, keepdims=True))

    def jac_g(self, covar: NDArray[np.float64]) -> NDArray[np.float64]:
        return covar * self.g(covar)


# Ts type var is tuple[NDArray[np.float64], tuple[NDArray[np.float64], ...]]
# first element type is NDArray[np.float64] and any other args are NDArray[np.float64]
class Regression(
    ParametricLifetimeModel[NDArray[np.float64], *tuple[NDArray[np.float64], ...]],
    ABC,
):
    def __init__(
        self,
        baseline: ParametricLifetimeModel[*tuple[NDArray[np.float64], ...]],
        coef: tuple[float, ...] | tuple[None] = (None,),
    ):
        super().__init__()
        self.compose_with(
            covar_effect=CovarEffect(coef),
            baseline=baseline,
        )

    def init_params(self, lifetimes: LifetimeSample) -> None:
        self.covar_effect.new_params(
            **{f"coef_{i}": 0.0 for i in range(lifetimes.rlc.args[0].shape[-1])}
        )
        self.baseline.init_params(lifetimes)

    @property
    def params_bounds(self) -> Bounds:
        lb = np.concatenate(
            (
                np.full(self.covar_effect.nb_params, -np.inf),
                self.baseline.params_bounds.lb,
            )
        )
        ub = np.concatenate(
            (
                np.full(self.covar_effect.nb_params, np.inf),
                self.baseline.params_bounds.ub,
            )
        )
        return Bounds(lb, ub)

    def sf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return super().sf(time, covar, *args)

    def isf(
        self,
        probability: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, covar, *args)

    def cdf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return super().cdf(time, covar, *args)

    def pdf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return super().pdf(time, covar, *args)

    def ppf(
        self,
        probability: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return super().ppf(probability, covar, *args)

    def mrl(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return super().mrl(time, covar, *args)

    def rvs(
        self,
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ):
        return super().rvs(covar, *args, size=size, seed=seed)

    def mean(
        self, covar: NDArray[np.float64], *args: * tuple[NDArray[np.float64], ...]
    ) -> NDArray[np.float64]:
        return super().mean(covar, *args)

    def var(
        self, covar: NDArray[np.float64], *args: * tuple[NDArray[np.float64], ...]
    ) -> NDArray[np.float64]:
        return super().var(covar, *args)

    def median(
        self, covar: NDArray[np.float64], *args: * tuple[NDArray[np.float64], ...]
    ) -> NDArray[np.float64]:
        return super().median(covar, *args)

    @abstractmethod
    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def jac_hf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def jac_chf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def dhf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]: ...

    # @property
    # def support_lower_bound(self):
    #     return 0.0
    #
    # @property
    # def support_upper_bound(self):
    #     return np.inf


class ProportionalHazard(Regression):

    def hf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.hf(time, *args)

    def chf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.chf(time, *args)

    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return self.baseline.ichf(
            cumulative_hazard_rate / self.covar_effect.g(covar), *args
        )

    def jac_hf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return np.column_stack(
            (
                self.covar_effect.jac_g(covar) * self.baseline.hf(time, *args),
                self.covar_effect.g(covar) * self.baseline.jac_hf(time, *args),
            )
        )

    def jac_chf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return np.column_stack(
            (
                self.covar_effect.jac_g(covar) * self.baseline.chf(time, *args),
                self.covar_effect.g(covar) * self.baseline.jac_chf(time, *args),
            )
        )

    def dhf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.dhf(time, *args)


class AFT(Regression):

    def hf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.hf(t0, *args) / self.covar_effect.g(covar)

    def chf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.chf(t0, *args)

    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.ichf(
            cumulative_hazard_rate, *args
        )

    def jac_hf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return np.column_stack(
            (
                -self.covar_effect.jac_g(covar)
                / self.covar_effect.g(covar) ** 2
                * (self.baseline.hf(t0, *args) + t0 * self.baseline.dhf(t0, *args)),
                self.baseline.jac_hf(t0, *args) / self.covar_effect.g(covar),
            )
        )

    def jac_chf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return np.column_stack(
            (
                -self.covar_effect.jac_g(covar)
                / self.covar_effect.g(covar)
                * t0
                * self.baseline.hf(t0, *args),
                self.baseline.jac_chf(t0, *args),
            )
        )

    def dhf(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: * tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.dhf(t0, *args) / self.covar_effect.g(covar) ** 2
