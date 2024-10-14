from abc import ABC, abstractmethod
from typing import Generic, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

DArgs = TypeVarTuple("DArgs")


class Discounting(Generic[*DArgs], ABC):
    @abstractmethod
    def factor(
        self, time: NDArray[np.float64], *args: *DArgs
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def rate(self, time: NDArray[np.float64], *args: *DArgs) -> NDArray[np.float64]: ...

    @abstractmethod
    def annuity_factor(
        self, time: NDArray[np.float64], *args: *DArgs
    ) -> NDArray[np.float64]: ...


class ExponentialDiscounting(Discounting[float | NDArray[np.float64]]):
    def factor(
        self,
        time: NDArray[np.float64],
        rate: float | NDArray[np.float64] = 0.0,
    ) -> NDArray[np.float64]:
        return np.exp(-rate * time)

    def rate(
        self,
        time: NDArray[np.float64],
        rate: float | NDArray[np.float64] = 0.0,
    ) -> NDArray[np.float64]:
        return rate * np.ones_like(time)

    def annuity_factor(
        self,
        time: NDArray[np.float64],
        rate: float | NDArray[np.float64] = 0.0,
    ) -> NDArray[np.float64]:
        mask = rate == 0
        rate = np.ma.MaskedArray(rate, mask)
        return np.where(mask, time, (1 - np.exp(-rate * time)) / rate)


class HyperbolicDiscounting(Discounting[float | NDArray[np.float64]]):

    def factor(
        self,
        time: NDArray[np.float64],
        beta: float | NDArray[np.float64] = 0.0,
    ) -> NDArray[np.float64]:
        return 1 / (1 + beta * time)

    def rate(
        self,
        time: NDArray[np.float64],
        beta: float | NDArray[np.float64] = 0.0,
    ) -> NDArray[np.float64]:
        return beta / (1 + beta * time)

    def annuity_factor(
        self,
        time: NDArray[np.float64],
        beta: float | NDArray[np.float64] = 0.0,
    ) -> NDArray[np.float64]:
        mask = beta == 0
        beta = np.ma.MaskedArray(beta, mask)
        return np.where(mask, time, np.log1p(beta * time) / beta)


class GeneralizedHyperbolicDiscounting(
    Discounting[float | NDArray[np.float64], float | NDArray[np.float64]]
):

    def factor(
        self,
        time: NDArray[np.float64],
        beta: float | NDArray[np.float64] = 0.0,
        eta: float | NDArray[np.float64] = 1.0,
    ) -> NDArray[np.float64]:
        return 1 / (1 + beta * time) ** eta

    def rate(
        self,
        time: NDArray[np.float64],
        beta: float | NDArray[np.float64] = 0.0,
        eta: float | NDArray[np.float64] = 1.0,
    ) -> NDArray[np.float64]:
        return beta * eta / (1 + beta * time)

    def annuity_factor(
        self,
        time: NDArray[np.float64],
        beta: float | NDArray[np.float64] = 0.0,
        eta: float | NDArray[np.float64] = 1.0,
    ) -> NDArray[np.float64]:
        mask_beta = beta == 0
        mask_eta = eta == 1
        beta = np.ma.MaskedArray(beta, mask_beta)
        eta = np.ma.MaskedArray(eta, mask_eta)
        return np.where(
            mask_eta,
            HyperbolicDiscounting.annuity_factor(time, beta),
            np.where(
                mask_beta,
                time,
                ((1 + beta * time) ** (1 - eta) - 1) / (beta * (1 - eta)),
            ),
        )


exponential_discounting = ExponentialDiscounting()
hyperbolic_discounting = HyperbolicDiscounting()
generalized_hyperbolic_discounting = GeneralizedHyperbolicDiscounting()
