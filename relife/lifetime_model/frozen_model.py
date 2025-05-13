from __future__ import annotations

from typing import TypeVarTuple, Generic, Optional, Callable
from typing_extensions import override

import numpy as np
from numpy.typing import NDArray

from relife.lifetime_model import ParametricLifetimeModel, LifetimeRegression


def _check_in_shape(name: str, value: float | NDArray[np.float64], nb_assets: int):
    value = np.asarray(value)
    if value.ndim > 2:
        raise ValueError
    if value.ndim == 2:
        value_nb_assets = value.shape[0]
        if nb_assets != 1:
            if value_nb_assets != 1 and value_nb_assets != nb_assets:
                raise ValueError(
                    f"Incorrect {name} shape. Got {value.shape}, meaning {value_nb_assets} nb_assets but args have {nb_assets} nb assets"
                )
    return value


Args = TypeVarTuple("Args")


class FrozenParametricLifetimeModel(ParametricLifetimeModel[()], Generic[*Args]):

    def __init__(self, model: ParametricLifetimeModel[*Args]):
        super().__init__()
        self.baseline = model
        self._args = ()
        self._nb_assets = 1

    @property
    def args(self) -> tuple[*Args]:
        return self._args

    @property
    def nb_assets(self) -> int:
        return self._nb_assets

    def unfreeze(self) -> ParametricLifetimeModel[*Args]:
        return self.baseline

    def freeze_args(self, *args: *Args):
        args_names = self.baseline.args_names
        if len(args) != len(args_names):
            raise ValueError(f"Expected {args_names} positional arguments but got only {len(args)} arguments")
        for name, value in zip(args_names, args):
            value = np.asarray(value)
            value: NDArray[np.float64]
            ndim = value.ndim
            if ndim > 2:
                raise ValueError(f"Uncorrect number of dimensions for {name}. It can't be higher than 2. Got {ndim}")
            match name:
                case "covar":  #  (), (nb_coef,) or (m, nb_coef)
                    if value.ndim == 2:  #  otherwise, when 1, broadcasting
                        if self.nb_assets != 1:
                            if value.shape[0] != self.nb_assets and value.shape[0] != 1:
                                raise ValueError(
                                    f"Invalid {name} values. Given {name} have {value.shape[0]} nb assets but other args gave {self.nb_assets} nb assets"
                                )
                        self._nb_assets = value.shape[0]  #  update nb_assets
                case "a0" | "ar" | "ar1" | "cf" | "cp" | "cr":
                    if value.ndim >= 1:
                        if self.nb_assets != 1:
                            if value.shape[0] != self.nb_assets and value.shape[0] != 1:
                                raise ValueError(
                                    f"Invalid {name} values. Given {name} have {value.shape[0]} nb assets but other args gave {self.nb_assets} nb assets"
                                )
                        self._nb_assets = value.shape[0]  #  update nb_assets
                case _:
                    raise ValueError(f"Unknown arg {name}")
            self._args = self.args + (value,)

    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.hf(time, *self.args)

    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.chf(time, *self.args)

    def sf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.sf(time, *self.args)

    def pdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.pdf(time, *self.args)

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.mrl(time, *self.args)

    @override
    def moment(self, n: int) -> np.float64 | NDArray[np.float64]:
        return self.baseline.moment(n, *self.args)

    @override
    def mean(self) -> np.float64 | NDArray[np.float64]:
        return self.baseline.moment(1, *self.args)

    @override
    def var(self) -> np.float64 | NDArray[np.float64]:
        return self.baseline.moment(2, *self.args) - self.baseline.moment(1, *self.args) ** 2

    @override
    def isf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        probability = _check_in_shape("probability", probability, self.args_nb_assets)
        return self.baseline.isf(probability, *self.args)

    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        cumulative_hazard_rate = _check_in_shape("cumulative_hazard_rate", cumulative_hazard_rate, self.args_nb_assets)
        return self.baseline.ichf(cumulative_hazard_rate, *self.args)

    @override
    def cdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.cdf(time, *self.args)

    @override
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int] = 1,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[int] = None,
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]
        | tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]
    ):
        return self.baseline.rvs(*self.args, size=size, return_event=return_event, return_entry=return_entry, seed=seed)

    @override
    def ppf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        probability = _check_in_shape("probability", probability, self.args_nb_assets)
        return self.baseline.ppf(probability, *self.args)

    @override
    def median(self) -> np.float64 | NDArray[np.float64]:
        return self.baseline.median(*self.args)

    @override
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 10,
    ) -> np.float64 | NDArray[np.float64]:
        a = _check_in_shape("a", a, self.args_nb_assets)
        b = _check_in_shape("b", b, self.args_nb_assets)
        return self.baseline.ls_integrate(func, a, b, *self.args, deg=deg)


class FrozenLifetimeRegression(FrozenParametricLifetimeModel[float | NDArray[np.float64], *Args]):

    baseline: LifetimeRegression[*Args]

    @override
    def freeze_args(self, covar: float | NDArray[np.float64], *args: *Args):
        super().freeze_args(*(covar, *args))

    @property
    def nb_coef(self) -> int:
        return self.baseline.nb_coef

    def dhf(
        self,
        time: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.dhf(time, self.args[0], *self.args[1:])

    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.jac_hf(time, self.args[0], *self.args[1:], asarray=asarray)

    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.jac_chf(time, self.args[0], *self.args[1:], asarray=asarray)

    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.jac_sf(time, self.args[0], *self.args[1:], asarray=asarray)

    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.jac_cdf(time, self.args[0], *self.args[1:], asarray=asarray)

    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.jac_pdf(time, self.args[0], *self.args[1:], asarray=asarray)
