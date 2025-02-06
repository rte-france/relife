from typing import Callable, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

MIN_POSITIVE_FLOAT = np.finfo(float).resolution


Ts = TypeVarTuple("Ts")


def gauss_legendre(
    func: Callable[[NDArray[np.float64], *Ts], NDArray[np.float64]],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *args: *Ts,
    ndim: int = 0,
    deg: int = 100,
) -> NDArray[np.float64]:
    shape = (-1,) + (1,) * ndim
    x, w = np.polynomial.legendre.leggauss(deg)
    x, w = x.reshape(shape), w.reshape(shape)
    p = (b - a) / 2
    m = (a + b) / 2
    u = p * x + m
    v = p * w
    return np.sum(v * func(u, *args), axis=0)


def gauss_laguerre(
    func: Callable[[NDArray[np.float64], *Ts], NDArray[np.float64]],
    *args: *Ts,
    ndim: int = 0,
    deg: int = 100,
) -> NDArray[np.float64]:
    shape = (-1,) + (1,) * ndim
    x, w = np.polynomial.laguerre.laggauss(deg)
    x, w = x.reshape(shape), w.reshape(shape)
    return np.sum(w * func(x, *args), axis=0)


def shifted_laguerre(
    func: Callable[[NDArray[np.float64], *Ts], NDArray[np.float64]],
    a: NDArray[np.float64],
    *args: *Ts,
    ndim: int = 0,
    deg: int = 100,
) -> NDArray[np.float64]:

    def f(x: NDArray[np.float64], *_: *Ts) -> NDArray[np.float64]:
        return func(x + a, *_) * np.exp(-a)

    return gauss_laguerre(f, *args, ndim=ndim, deg=deg)


def quad_laguerre(
    func: Callable[[NDArray[np.float64], *Ts], NDArray[np.float64]],
    a: NDArray[np.float64],
    *args: *Ts,
    ndim: int = 0,
    deg: int = 100,
) -> NDArray[np.float64]:

    def f(x: NDArray[np.float64], *_: *Ts) -> NDArray[np.float64]:
        return func(x + a, *_) * np.exp(x)

    return gauss_laguerre(f, *args, ndim=ndim, deg=deg)
