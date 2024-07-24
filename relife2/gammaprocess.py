from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from scipy.optimize import Bounds, bisect

import relife2.distributions as distributions
from relife2 import parametric
from relife2.types import FloatArray


class ShapeFunctions(parametric.Functions, ABC):
    """BLABLABLA"""

    def init_params(self, *args: Any) -> FloatArray:
        return np.ones_like(self.params)

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        return Bounds(
            np.full(self.params.size, np.finfo(float).resolution),
            np.full(self.params.size, np.inf),
        )

    @abstractmethod
    def nu(self, time: FloatArray) -> FloatArray:
        """BLABLABLA"""

    @abstractmethod
    def jac_nu(self, time: FloatArray) -> FloatArray:
        """BLABLABLA"""


class PowerShapeFunctions(ShapeFunctions):
    """BLABLABLA"""

    def __init__(
        self, shape_rate: Optional[float] = None, shape_power: Optional[float] = None
    ):
        super().__init__(shape_rate=shape_rate, shape_power=shape_power)

    def nu(self, time: FloatArray) -> FloatArray:
        return self.shape_rate * time**self.shape_power

    def jac_nu(self, time: FloatArray) -> FloatArray:
        return self.shape_rate * self.shape_power * time ** (self.shape_power - 1)


class ExponentialShapeFunctions(ShapeFunctions):
    """BLABLABLA"""

    def __init__(self, shape_exponent: Optional[float] = None):
        super().__init__(shape_exponent=shape_exponent)

    def nu(self, time: FloatArray) -> FloatArray:
        pass

    def jac_nu(self, time: FloatArray) -> FloatArray:
        pass


# GammaProcessFunctions(FunctionsBridge, parametric.Functions)
class GPFunctions(parametric.Functions):
    """BLABLABLA"""

    def __init__(
        self,
        shape_function: ShapeFunctions,
        rate: Optional[float] = None,
        initial_resistance: Optional[float] = None,
        load_threshold: Optional[float] = None,
    ):

        super().__init__()
        self.add_functions(
            "process_lifetime_distribution",
            distributions.GPDistributionFunctions(
                shape_function, rate, initial_resistance, load_threshold
            ),
        )

    def init_params(self, *args: Any) -> FloatArray:
        return self.process_lifetime_distribution.init_params(*args)

    @property
    def params_bounds(self) -> Bounds:
        return self.process_lifetime_distribution.params_bounds

    def _g(
        self, times, times_after_failure, times_before_failure, deteriorations_before, u
    ):
        return (
            1
            / (
                1
                - self.process_lifetime_distribution.conditional_sf(
                    times_after_failure,
                    times_before_failure,
                    deteriorations_before,
                )
            )
        ) * (
            self.process_lifetime_distribution.conditional_sf(
                times, times_before_failure, deteriorations_before
            )
            - self.process_lifetime_distribution.conditional_sf(
                times_after_failure,
                times_before_failure,
                deteriorations_before,
            )
        ) - u

    def sample(
        self,
        inspection_times: FloatArray,
        unit_ids=None,
        nb_sample=1,
        seed=None,
        add_death_time=True,
    ) -> FloatArray:
        """
        inspection_times has 0 at first
        inspection_times is sorted with unit_ids ([0, 2, 3, 4, 5, 10, 0, 3, 6, 11,...], [0, 0, 0, 0, 0, 0, 1, 1 ,1, 1, ...])
        N : nb_units
        M : nb_measures
        B : nb_sample

        inspection_times : (N*M,)
        unit_ids : (N*M,)
        """

        np.random.seed(seed)
        if unit_ids is None:
            unit_ids = np.zeros_like(inspection_times)

        unique_unit_ids, counts_of_ids = np.unique(
            unit_ids, return_counts=True
        )  # (N,) (N,)

        inspection_times = np.tile(inspection_times, nb_sample)  # (N*M*B,)
        unit_ids = np.tile(unit_ids, nb_sample)  # (N*M*B,)
        sample_ids = np.repeat(np.arange(nb_sample), np.sum(counts_of_ids))  # (N*M*B,)
        inc_credits = np.tile(counts_of_ids, nb_sample) - 1  # (N*B, )
        # (N*B,)
        # init
        # index = np.cumsum(np.insert(counts_of_ids, 0, 1))[:-1] - 1  # (N*B,)
        index = np.where(inspection_times == 0)[0]
        times = inspection_times[index]  # (N*B,)
        nu = self.process_lifetime_distribution.shape_function.nu(times)  # (N*B,)
        # increments = np.random.gamma(nu, 1 / self.rate)  # (N*B,)
        increments = np.zeros_like(nu)
        deteriorations = (
            self.process_lifetime_distribution.initial_resistance - increments
        )  # (N*B,)

        # counts_of_ids = counts_of_ids - 1  # (N*B,)

        cond = np.logical_and(
            deteriorations > self.process_lifetime_distribution.load_threshold,
            inc_credits > 0,
        )  # (N*B,)

        res_deteriorations = deteriorations.copy()  # (N*B,)
        res_times = times.copy()  # (N*B,)
        res_unit_ids = unit_ids[index]  # (N*B,)
        res_sample_ids = sample_ids[index]  # (N*B,)

        while cond.any():

            # next values
            next_index = index[cond] + 1
            next_times = inspection_times[next_index]
            next_nu = self.process_lifetime_distribution.shape_function.nu(next_times)

            try:
                increments = np.random.gamma(next_nu - nu[cond], 1 / self.rate)
            except ValueError:
                print("putain de merde")

            deteriorations = deteriorations[cond] - increments

            if add_death_time:
                index_of_failures = (
                    deteriorations < self.process_lifetime_distribution.load_threshold
                )

                nb_of_failures = np.sum(index_of_failures)
                if nb_of_failures > 0:
                    u = np.random.uniform(0, 1, size=nb_of_failures)
                    times_before_failure = inspection_times[next_index - 1][
                        index_of_failures
                    ]
                    times_after_failure = next_times[index_of_failures]
                    deteriorations_before = (deteriorations + increments)[
                        index_of_failures
                    ]

                    failure_times = np.array(
                        [
                            bisect(
                                self._g,
                                a,
                                b,
                                args=(
                                    b,
                                    a,
                                    d,
                                    _u,
                                ),
                            )
                            for (a, b, d, _u) in zip(
                                times_before_failure,
                                times_after_failure,
                                deteriorations_before,
                                u,
                            )
                        ]
                    )
                    deteriorations[index_of_failures] = (
                        self.process_lifetime_distribution.load_threshold
                    )
                    next_times[index_of_failures] = failure_times

                    # try:
                    assert (failure_times < times_after_failure).all()
                    # except AssertionError:
                    #     print(failure_times)
                    #     print(times_after_failure)
                    #     print(times_before_failure)
                    #     print(deteriorations_before)
                    #     print(u)
                    #     sys.exit()
                    assert (failure_times > times_before_failure).all()

            res_deteriorations = np.concatenate([res_deteriorations, deteriorations])
            res_times = np.concatenate([res_times, next_times])
            res_unit_ids = np.concatenate([res_unit_ids, unit_ids[next_index]])
            res_sample_ids = np.concatenate([res_sample_ids, sample_ids[next_index]])

            inc_credits = inc_credits[cond] - 1
            # inc_credits[index_of_failures] = 0
            cond = np.logical_and(
                deteriorations > self.process_lifetime_distribution.load_threshold,
                inc_credits > 0,
            )  # (N*B,)

            # update
            index = next_index.copy()
            nu = next_nu.copy()

        return res_deteriorations, res_times, res_unit_ids, res_sample_ids
