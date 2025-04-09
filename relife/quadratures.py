from typing import Callable, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from relife._args import get_nb_assets
from relife.lifetime_model import FrozenParametricLifetimeModel

MIN_POSITIVE_FLOAT = np.finfo(float).resolution


def reshape_bound(x : float | NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.asarray(x)
    if x.ndim > 2:
        raise ValueError
    if x.size > 1:
        return x.reshape(-1, 1)
    return x

def gauss_legendre(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float|NDArray[np.float64],
    b: float|NDArray[np.float64],
    ndim : int = 1,
    deg: int = 100,
) -> NDArray[np.float64]:
    if ndim > 2:
        raise ValueError
    shape = (-1) + (1,) * ndim
    x, w = np.polynomial.legendre.leggauss(deg)
    x, w = x.reshape(shape), w.reshape(shape)
    p = (b - a) / 2
    m = (a + b) / 2
    u = p * x + m
    v = p * w
    return np.sum(v * func(u), axis=0)


def gauss_laguerre(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ndim : int = 1,
    deg: int = 100,
) -> NDArray[np.float64]:
    if ndim > 2:
        raise ValueError
    shape = (-1) + (1,) * ndim
    x, w = np.polynomial.laguerre.laggauss(deg)
    x, w = x.reshape(shape), w.reshape(shape)
    return np.sum(w * func(x), axis=0)


def shifted_laguerre(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: NDArray[np.float64],
    ndim: int = 1,
    deg: int = 100,
) -> NDArray[np.float64]:
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return func(x + a) * np.exp(-a)

    return gauss_laguerre(f, ndim=ndim, deg=deg)


def quad_laguerre(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64],
    ndim : int = 1,
    deg: int = 100,
) -> NDArray[np.float64]:
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return func(x + a) * np.exp(x)
    return gauss_laguerre(f, ndim=ndim, deg=deg)



def ls_integrate(
    model : FrozenParametricLifetimeModel,
    func : Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
    a : float | NDArray[np.float64],
    b : float | NDArray[np.float64],
    deg : int = 100
):
    from relife.lifetime_model import FrozenParametricLifetimeModel, LifetimeDistribution, LifetimeRegression, AgeReplacementModel

    if not isinstance(model, FrozenParametricLifetimeModel):
        raise ValueError

    a = reshape_bound(a)
    b = reshape_bound(b)
    if get_nb_assets(a) != get_nb_assets(b) != get_nb_assets(*model.args):
        raise ValueError
    a, b = np.broadcast_to(a, b)

    match model.model:
        case LifetimeDistribution() | LifetimeRegression():
            b = np.minimum(np.inf, b)
            def integrand(x: float | NDArray[np.float64]) -> NDArray[np.float64]:
                return func(x) * model.pdf(x)

            if np.all(np.isinf(b)):
                b = np.atleast_2d(model.isf(np.array(1e-4)))

            return np.where(
                np.isinf(b),
                gauss_legendre(integrand, a, b, ndim=model.ndim, deg=deg) + quad_laguerre(integrand, b, ndim=model.ndim, deg=deg),
                gauss_legendre(integrand, a, b, ndim=model.ndim, deg=deg)
            )

        case AgeReplacementModel():
            ar = model.args[0]
            ub = np.minimum(np.inf, ar)
            b = np.minimum(ub, b)
            def integrand(x: float | NDArray[np.float64]) -> NDArray[np.float64]:
                return np.atleast_2d(func(x) * model.pdf(x))
            w = np.where(b == ar, func(ar) * model.sf(ar), 0)
            return gauss_legendre(integrand, a, b, ndim=model.ndim, deg=deg) + w






