from dataclasses import dataclass
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sc
from scipy.optimize import minimize
from scipy.special import gamma, gammainc

from relife.model import AbsolutelyContinuousLifetimeModel
from .utils import gauss_legendre, quad_laguerre, moore_jac_uppergamma_c

matplotlib.use('Qt5Agg', force=True)


@dataclass
class GammaProcessData:  # stock les données réelles ou en simule

    path: np.array = None
    increments: np.array = None
    inspection_times: np.array = None
    r0: float = None
    l0: float = None
    scale: float = None
    shape_scaling: float = None
    shape_power: float = None
    nb_sample: int = 1
    declared_data: bool = None
    simulated_data: bool = None

    def valid_inputs(self) -> None:

        simulation_params = (self.r0, self.l0, self.scale, self.shape_scaling, self.shape_power)

        if self.inspection_times is None:
            raise ValueError("'Inspection times' is required")

        if (self.path is None) and all(x is None for x in simulation_params):
            raise ValueError("'Path' xor simulation parameters are required")

        if (self.path is not None) and not all(x is None for x in simulation_params):
            raise ValueError("'Path' and simulation parameters must not be both declared")

        if (self.path is None) and any(x is None for x in simulation_params):
            raise ValueError("All simulation parameters must be declared")

        if all(x is None for x in simulation_params) and any(x is None for x in self.path):
            raise ValueError("'Inspection times' and 'path' must be declared")

        if all(x is None for x in simulation_params) and (np.size(self.inspection_times) != np.size(self.path)):
            raise ValueError("'Inspection times' and 'path' must have the same length")

        if all(x is None for x in simulation_params) and (np.size(self.inspection_times) == 1):
            raise ValueError("'Inspection times' and 'path' must contain at least two data points")

        if all(x is None for x in simulation_params) and any(self.inspection_times < 0):
            raise ValueError("'Inspection times' must be positive")

        if all(x is None for x in simulation_params) and any(np.diff(self.path) < 0):
            raise ValueError("'Path' must contain increasing values")

        if all(x is None for x in simulation_params) and any(np.diff(self.inspection_times) < 0):
            raise ValueError("'Inspection times' must contain increasing values")

        if (self.path is None) and any([np.size(x) > 1 for x in simulation_params]):
            raise ValueError("Simulation parameters must have length 1")

        if (self.path is None) and any([x < 0 for x in simulation_params]):
            raise ValueError("Simulation parameters must be positive")

        if self.path is not None:
            self.declared_data = True
            self.simulated_data = False
            self.increments = np.diff(self.path)

        if all(x is not None for x in simulation_params):
            self.declared_data = False
            self.simulated_data = True

    def sample_path(self):
        self.valid_inputs()
        if self.declared_data:
            raise ValueError("No sample path to generate because 'path' is already declared.")

        def v(t):
            return self.shape_scaling * t ** self.shape_power

        n = len(self.inspection_times) - 1
        h = np.diff(self.inspection_times)
        shape = np.repeat((v(self.inspection_times[:-1] + h) - v(self.inspection_times[:-1])).reshape(1, -1),
                          self.nb_sample,
                          axis=0)
        inc = np.random.gamma(shape, 1 / self.scale, (self.nb_sample, n))
        X = np.insert(np.cumsum(inc, axis=1), 0, 0, axis=1)
        self.path = X
        self.increments = inc

    def __post_init__(self):
        self.valid_inputs()
        print('valid_input() was executed')
        if self.simulated_data:
            self.sample_path()
            print('sample_path() was executed')


        # return GammaProcessData(self.nb_sample, self.inspection_times, inc, X)


class GammaProcessLifetimeModel(AbsolutelyContinuousLifetimeModel):

    def __init__(self, r0, l0, scale, shape_scaling, shape_power):
        self.r0 = r0
        self.l0 = l0
        self.scale = scale
        self.shape_scaling = shape_scaling
        self.shape_power = shape_power

    def shape_function(self, t):
        return self.shape_scaling * t ** self.shape_power

    def support_upper_bound(self, *args: np.ndarray) -> float:
        return np.inf

    def sf(self, t):
        return sc.gammainc(self.shape_function(t), (self.r0 - self.l0) * self.scale)

    def pdf(self, t):
        return self.shape_power / t * self.shape_function(t) * (
                moore_jac_uppergamma_c(self.shape_function(t), (self.r0 - self.l0) * self.scale) / sc.gamma(
            self.shape_function(t)) - sc.digamma(self.shape_function(t)) * (
                        1 - sc.gammainc(self.shape_function(t), (self.r0 - self.l0) * self.scale)))

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


class GammaProcess():

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
