import numpy as np
import scipy.special as sc
from scipy.special import gamma, gammainc
from scipy.optimize import minimize, newton
import matplotlib
import matplotlib.pyplot as plt
from relife.model import HazardFunctions, AbsolutelyContinuousLifetimeModel
from typing import Callable
from .utils import gauss_legendre, quad_laguerre, moore_jac_uppergamma_c

matplotlib.use('Qt5Agg', force=True)


class GammaProcessData:

    def __init__(self, inspection_times, increments, paths, nb_sample):
        self.inspection_times = inspection_times
        self.increments = increments
        self.paths = paths
        self.nb_sample = nb_sample


class GammaProcessLifetimeModel(AbsolutelyContinuousLifetimeModel):

    def __init__(self, r0, l0, scale, c, b):
        self.r0 = r0
        self.l0 = l0
        self.scale = scale
        self.c = c
        self.b = b

    def v(self, t):
        return self.c * t ** self.b

    def support_upper_bound(self, *args: np.ndarray) -> float:
        return np.inf

    def sf(self, t):
        return sc.gammainc(self.v(t), (self.r0 - self.l0) * self.scale)

    def pdf(self, t):
        return self.b / t * self.v(t) * (moore_jac_uppergamma_c(self.v(t), (self.r0 - self.l0) * self.scale) / sc.gamma(
            self.v(t)) - sc.digamma(self.v(t)) * (1 - sc.gammainc(self.v(t), (self.r0 - self.l0) * self.scale)))

    def hf(self, t):
        return self.pdf(t) / self.sf(t)

    def chf(self, t):
        return -np.log(self.sf(t))

    def ichf(self, v: np.ndarray, *args: np.ndarray) -> np.ndarray:
        initial_guess = np.ones_like(v)
        return sc.optimize.newton(func=lambda t: self.chf(t) - v, x0=initial_guess, fprime=lambda t: self.hf(t))

    def isf(self, p: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.ichf(-np.log(p), *args)

    def ls_integrate(
            self,
            func: Callable,
            a: np.ndarray,
            b: np.ndarray,
            *args: np.ndarray,
            ndim: int = 0,
            deg: int = 100,
            q0: float = 1e-4
    ) -> np.ndarray:
        ub = self.support_upper_bound(*args)
        b = np.minimum(ub, b)
        f = lambda x, *args: func(x) * self.pdf(x, *args)
        if np.all(np.isinf(b)):
            b = self.isf(q0, *args)
            res = quad_laguerre(f, b, *args, ndim=ndim, deg=deg)
        else:
            res = 0
        return gauss_legendre(f, a, b, *args, ndim=ndim, deg=deg) + res


class GammaProcess(HazardFunctions):

    def __init__(self, scale_parameter, shape_scaling, shape_power):
        self.shape_power = shape_power
        self.shape_scaling = shape_scaling
        self.scale_parameter = scale_parameter
        self.nb_sample = None
        self.paths = None
        self.increments = None
        self.inspection_times = None

    def shape_function(self, t):
        return self.shape_scaling * t ** self.shape_power

    def sample_path(self, inspection_times, nb_sample=1):
        n = len(inspection_times) - 1
        h = np.diff(inspection_times)
        shape = np.repeat((self.shape_function(inspection_times[:-1] + h)
                           - self.shape_function(inspection_times[:-1])).reshape(1, -1),
                          nb_sample,
                          axis=0)
        inc = np.random.gamma(shape, 1 / self.scale_parameter, (nb_sample, n))
        X = np.insert(np.cumsum(inc, axis=1), 0, 0, axis=1)
        self.paths = X
        self.increments = inc
        self.inspection_times = inspection_times
        self.nb_sample = nb_sample
        return GammaProcessData(nb_sample, inspection_times, inc, X)

    def negative_log_likelihood(self, params, sample_id):
        # TODO:
        #   - self.increments contient potentiellement des valeurs nulles, ce qui renvoie une likelihood infinie. Que
        #   faire dans ce cas ? Comment le prévenir ?

        if sample_id is None:
            sample_id = 0

        shape_scaling, shape_power, scale_parameter = params
        t, delta = self.inspection_times, self.increments[sample_id]
        t_plus = t[1:]
        t_minus = t[:-1]
        return -np.sum(
            shape_scaling * (t_plus ** shape_power - t_minus ** shape_power) * np.log(scale_parameter)
            - np.log(gamma(shape_scaling * (t_plus ** shape_power - t_minus ** shape_power)))
            + (shape_scaling * (t_plus ** shape_power - t_minus ** shape_power) - 1)
            * np.log(delta) - scale_parameter * delta
        )

    def method_of_moments(self):
        pass

    def fit(self, sample_id=None):
        # TODO:
        #   - Comment initialiser la recherche de l'optimum?
        if sample_id is None:
            sample_id = np.arange(self.nb_sample)
        else:
            sample_id = np.array(sample_id)

        optimums = np.empty((len(sample_id), 3))
        k = 0
        for i in sample_id:
            opt = minimize(
                fun=self.negative_log_likelihood,
                x0=np.array([1, 1, 1]),
                args=i,
                method='Nelder-Mead'
            )
            optimums[k,] = opt.x
            k += 1
        return optimums


class GammaProcessDeterministicLoadsModel:

    def __init__(self, r0, l0, scale_parameter, shape_scaling, shape_power):
        self.r0 = r0
        self.l0 = l0
        self.shape_power = shape_power
        self.shape_scaling = shape_scaling
        self.scale_parameter = scale_parameter

    def shape_function(self, t):
        return self.shape_scaling * t ** self.shape_power

    def theoretical_survival_function(self, t):
        return gammainc(self.shape_function(t), (self.r0 - self.l0) * self.scale_parameter)

    def sample_path(self, inspection_times, nb_sample=1):
        h = np.diff(inspection_times)
        shape = np.repeat((self.shape_function(inspection_times[:-1] + h)
                           - self.shape_function(inspection_times[:-1])).reshape(1, -1),
                          nb_sample,
                          axis=0)
        inc = np.random.gamma(shape, 1 / self.scale_parameter, (nb_sample, n))
        X = np.insert(np.cumsum(inc, axis=1), 0, 0, axis=1)
        self.paths = X
        self.increments = inc
        self.inspection_times = inspection_times
        self.nb_sample = nb_sample
        return GammaProcessData(nb_sample, inspection_times, inc, X)

    def lifetime_sample(self, n):
        pass

    def plot(self):
        pass


class RandomLoadsModel:

    def func(self):
        pass


if __name__ == "__main__":
    ##############
    # PARAMETERS #
    ##############
    T = 100
    N = 1000
    SCALE_PARAMETER = 10
    SHAPE_SCALING = 5
    SHAPE_POWER = 1 / 2
    NB_SAMPLE = 10
    SAMPLE_ID = [2, 3, 4]
    PARAMS = (SHAPE_SCALING, SHAPE_POWER, SCALE_PARAMETER)
    INSPECTION_TIMES = np.linspace(start=0, stop=T, num=N)

    GP = GammaProcess(SCALE_PARAMETER, SHAPE_SCALING, SHAPE_POWER)
    GP.sample_path(inspection_times=INSPECTION_TIMES, nb_sample=NB_SAMPLE)
    print(np.min(GP.increments[0]))
    print(GP.fit(SAMPLE_ID))
    print(PARAMS)
    plt.plot(GP.inspection_times, GP.paths[0])
    plt.show()
