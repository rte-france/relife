"""
This module defines probability functions used in gamma process

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from random import uniform
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, bisect, minimize
from scipy.special import digamma, expi, gammainc, lambertw, loggamma
from scipy.stats import gamma

from relife2.core import (
    Likelihood,
    ParametricComponent,
    ParametricLifetimeModel,
    ParametricModel,
)
from relife2.data import Deteriorations, deteriorations_factory
from relife2.io import array_factory


class ShapeFunctions(ParametricComponent, ABC):
    """BLABLABLA"""

    def init_params(self, *args: Any) -> NDArray[np.float64]:
        return np.ones_like(self.params)

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        return Bounds(
            np.full(self.params.size, np.finfo(float).resolution),
            np.full(self.params.size, np.inf),
        )

    @abstractmethod
    def nu(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        """BLABLABLA"""

    @abstractmethod
    def jac_nu(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        """BLABLABLA"""


class PowerShape(ShapeFunctions):
    """BLABLABLA"""

    def __init__(
        self, shape_rate: Optional[float] = None, shape_power: Optional[float] = None
    ):
        super().__init__()
        self.new_params(shape_rate=shape_rate, shape_power=shape_power)

    def nu(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.shape_rate * time**self.shape_power

    def jac_nu(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.shape_rate * self.shape_power * time ** (self.shape_power - 1)


# class ExponentialShapeFunctions(ShapeFunctions):
#     """BLABLABLA"""
#
#     def __init__(self, shape_exponent: Optional[float] = None):
#         super().__init__(shape_exponent=shape_exponent)
#
#     def nu(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
#         return None
#
#     def jac_nu(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
#         return None


class GammaProcessDistribution(
    ParametricLifetimeModel[*tuple[NDArray[np.float64], NDArray[np.float64]]]
):
    """BLABLABLA"""

    def __init__(
        self,
        shape_function: ShapeFunctions,
        rate: Optional[float] = None,
    ):
        super().__init__()
        self.new_params(rate=rate)
        self.compose_with(shape_function=shape_function)
        self.extras["initial_resistance"] = uniform(1, 2)
        self.extras["load_threshold"] = uniform(0, 1)

    @property
    def support_lower_bound(self):
        """
        Returns:
            BLABLABLA
        """
        return 0.0

    @property
    def support_upper_bound(self):
        """
        Returns:
            BLABLABLA
        """
        return np.inf

    @property
    def resistance_magnitude(self):
        """
        Returns:
        """
        return (self.initial_resistance - self.load_threshold) * self.rate

    def init_params(self, *args: Any) -> NDArray[np.float64]:
        """
        Args:
            *args ():
        Returns:
        """
        return np.concatenate(
            (
                np.array([1]),
                self.shape_function.init_params(*args),
            )
        )

    @property
    def params_bounds(self) -> Bounds:
        lb = np.concatenate(
            (
                np.array([np.finfo(float).resolution]),
                self.shape_function.params_bounds.lb,
            )
        )
        ub = np.concatenate(
            (
                np.array([np.inf]),
                self.shape_function.params_bounds.ub,
            )
        )
        return Bounds(lb, ub)

    def pdf(
        self,
        time: NDArray[np.float64],
        l0: NDArray[np.float64],
        r0: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """BLABLABLA"""

        res = -self.shape_function.jac_nu(time) * self.moore_jac_uppergamma_c(time)

        return np.where(
            time == 0,
            int(self.shape_power == 1)
            * (-self.shape_rate * expi(-self.resistance_magnitude)),
            res,
        )

    def sf(
        self,
        time: NDArray[np.float64],
        l0: NDArray[np.float64],
        r0: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        BLABLABLABLA
        Args:
            time (NDArray[np.float64]): BLABLABLABLA

        Returns:
            Union[float, NDArray[np.float64]]: BLABLABLABLA
        """
        return gammainc(
            self.shape_function.nu(time),
            (self.initial_resistance - self.load_threshold) * self.rate,
        )

    def conditional_sf(
        self,
        time: NDArray[np.float64],
        conditional_time: NDArray[np.float64],
        conditional_resistance: NDArray[np.float64],
        l0: NDArray[np.float64],
        r0: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Args:
            time ():
            conditional_time ():
            conditional_resistance ():

        Returns:
        """

        return gammainc(
            self.shape_function.nu(time) - self.shape_function.nu(conditional_time),
            (conditional_resistance - self.load_threshold) * self.rate,
        )

    def _series_expansion(
        self, shape_values: NDArray[np.float64], tol: float
    ) -> NDArray[np.float64]:

        # resistance_values - shape : (n,)

        # shape : (n,)
        r = self.resistance_magnitude / (1 + shape_values)

        # shape : (n,)
        f = np.exp(
            shape_values * np.log(self.resistance_magnitude)
            - loggamma(shape_values + 1)
            - self.resistance_magnitude
        )
        # shape : (n,)
        d_f = f * (np.log(self.resistance_magnitude) - digamma(shape_values + 1))
        # shape : (n,)
        epsilon = tol / (abs(f) + abs(d_f))
        # shape : (n,)
        delta = (1 - r) * epsilon / 2

        # shape : (n,)
        n1 = np.ceil((np.log(epsilon) + np.log(1 - r)) / np.log(r)).astype(np.int64)
        n2 = np.ceil(1 + r / (1 - r)).astype(np.int64)

        # /!\ n3 does not run if condition ind not true
        n3 = np.ceil(np.real(lambertw(np.log(r) * delta, k=-1) / np.log(r))).astype(
            np.int64
        )

        # M = max(n1, n2, 3)
        # shape : (n, M)
        # print("n1 :", n1)
        # print("n2 :", n2)
        # print("n3 :", n3)
        # print("M :", np.max((n1, n2, n3), initial=0.0) + 1)
        range_grid = np.tile(
            np.arange(1, np.max((n1, n2, n3), initial=0.0) + 1),
            (len(n1), 1),
        )

        # print("range_grid :", range_grid)
        mask = np.ones_like(range_grid)

        # shape : (n,), fill range_grid with zeros when crossed max upper bound on conditioned indices
        ind = np.log(r) * delta >= -1 / np.exp(1)

        # print("ind :", ind)

        # shape : (n, M)
        mask[ind] = (
            range_grid[ind]
            <= np.maximum(np.max(np.vstack((n1, n2)), axis=0), n3)[ind][:, None]
        )
        # print(np.maximum(np.max(np.vstack((n1, n2)), axis=0), n3)[ind][:, None])
        mask[~ind] = (
            range_grid[~ind] <= np.max(np.vstack((n1, n2)), axis=0)[~ind][:, None]
        )
        # print(np.max(np.vstack((n1, n2)), axis=0)[~ind][:, None])
        # print("mask :", mask)

        # shape : (n, M + 1)
        harmonic = np.hstack(
            (
                np.zeros((range_grid.shape[0], 1)),
                1 / (range_grid + shape_values[:, None]),
            ),
        )
        cn = np.hstack(
            (
                np.ones((range_grid.shape[0], 1)),
                self.resistance_magnitude / (range_grid + shape_values[:, None]),
            ),
        )
        mask = np.hstack((np.ones((range_grid.shape[0], 1)), mask))

        # shape : (n, M + 1)
        harmonic = np.cumsum(harmonic, axis=1) * mask
        cn = np.cumprod(cn, axis=1) * mask

        cn_derivative = -cn * harmonic

        # shape : (n,)
        s = np.sum(cn, axis=1)
        d_s = np.sum(cn_derivative, axis=1)

        # return shape : (n,)
        return s * d_f + f * d_s

    def _continued_fraction_expansion(
        self, shape_values: NDArray[np.float64], tol: float
    ) -> NDArray[np.float64]:

        # resistance_values - shape : (n,)

        # shape : (n, 2)
        a = np.tile(
            np.array([1, 1 + self.resistance_magnitude]),
            (len(shape_values), 1),
        )
        b = np.hstack(
            (
                (np.ones_like(shape_values) * self.resistance_magnitude)[:, None],
                self.resistance_magnitude
                * (2 - shape_values[:, None] + self.resistance_magnitude),
            )
        )

        # shape : (n, 2)
        d_a = np.zeros_like(a)
        d_b = np.zeros_like(b)
        d_b[:, 1] = -self.resistance_magnitude

        # shape : (n,)
        f = np.exp(
            shape_values * np.log(self.resistance_magnitude)
            - loggamma(shape_values)
            - self.resistance_magnitude
        )
        d_f = f * (np.log(self.resistance_magnitude) - digamma(shape_values))

        s = None

        res = np.ones_like(shape_values) * 2 * tol
        k = 2

        result = np.full_like(shape_values, np.nan)
        d_result = result.copy()

        while (res > tol).any():

            ak = (k - 1) * (shape_values - k)
            bk = 2 * k - shape_values + self.resistance_magnitude

            next_a = bk * a[:, 1] + ak * a[:, 0]
            next_b = bk * b[:, 1] + ak * b[:, 0]

            next_d_a = bk * d_a[:, 1] - a[:, 1] + ak * d_a[:, 0] + (k - 1) * a[:, 0]
            next_d_b = bk * d_b[:, 1] - b[:, 1] + ak * d_b[:, 0] + (k - 1) * b[:, 0]

            next_s = next_a / next_b

            if s is not None:
                res = np.abs(next_s - s) / next_s
            k += 1

            # update
            a = np.hstack((a[:, [1]], next_a[:, None]))
            b = np.hstack((b[:, [1]], next_b[:, None]))
            d_a = np.hstack((d_a[:, [1]], next_d_a[:, None]))
            d_b = np.hstack((d_b[:, [1]], next_d_b[:, None]))
            result[res <= tol] = next_s[res <= tol]
            d_result[res <= tol] = (
                next_b ** (-2) * (next_b * next_d_a - next_a * next_d_b)
            )[res <= tol]
            s = next_s

        return -f * d_result - result * d_f

    def moore_jac_uppergamma_c(
        self, time: NDArray[np.float64], tol: float = 1e-6
    ) -> NDArray[np.float64]:
        """BLABLABLA"""

        # /!\ consider time as masked array
        shape_values = np.ravel(self.shape_function.nu(time))
        zero_time = np.ravel(time == 0)
        # print("shape_values :", shape_values)
        # print("zero_time", zero_time)

        series_indices = np.logical_or(
            np.logical_and(
                shape_values <= self.resistance_magnitude,
                self.resistance_magnitude <= 1,
            ),
            self.resistance_magnitude < shape_values,
        )
        # print("series_indices :", series_indices)
        # print(np.logical_and(~zero_time, series_indices))
        # print(shape_values[np.logical_and(~zero_time, series_indices)])

        result = np.full_like(shape_values, np.nan)
        result[np.logical_and(~zero_time, series_indices)] = self._series_expansion(
            shape_values[np.logical_and(~zero_time, series_indices)], tol
        )
        result[np.logical_and(~zero_time, ~series_indices)] = (
            self._continued_fraction_expansion(
                shape_values[np.logical_and(~zero_time, ~series_indices)], tol
            )
        )
        result[zero_time] = 0
        # print("result moore jac:", result.reshape(time.shape))

        return result.reshape(time.shape)


class LikelihoodFromDeteriorations(Likelihood):
    """BLABLABLA"""

    def __init__(
        self,
        functions: ParametricModel,
        deterioration_data: Deteriorations,
        first_increment_uncertainty: Optional[tuple] = None,
        measurement_tol: np.floating[Any] = np.finfo(float).resolution,
    ):
        super().__init__(functions)
        self.deterioration_data = deterioration_data
        self.first_increment_uncertainty = first_increment_uncertainty
        self.measurement_tol = measurement_tol

    def negative_log(self, params: NDArray[np.float64]) -> float:
        """
        All deteriorations have R0 in first column
        All times have 0 in first column
        """
        self.params = params

        delta_shape = np.diff(
            self.function.shape_function.nu(self.deterioration_data.times),
            axis=1,
        )

        contributions = -(
            delta_shape * np.log(self.rate)
            + (delta_shape - 1)
            * np.log(
                self.deterioration_data.increments,
                where=~self.deterioration_data.event,
                out=np.zeros_like(delta_shape),
            )
            - self.rate * self.deterioration_data.increments
            - np.log(
                gamma_function(delta_shape),
                where=~self.deterioration_data.event,
                out=np.zeros_like(delta_shape),
            )
        )

        censored_contributions = -np.log(
            gamma.cdf(
                self.deterioration_data.increments + self.measurement_tol,
                a=np.diff(
                    self.function.shape_function.nu(self.deterioration_data.times)
                ),
                scale=1 / self.rate,
            )
            - gamma.cdf(
                self.deterioration_data.increments - self.measurement_tol,
                a=np.diff(
                    self.function.shape_function.nu(self.deterioration_data.times)
                ),
                scale=1 / self.rate,
            ),
            where=self.deterioration_data.event,
            out=np.zeros_like(delta_shape),
        )

        contributions = np.where(
            self.deterioration_data.event, censored_contributions, contributions
        )

        if self.first_increment_uncertainty is not None:

            first_inspections = self.deterioration_data.times[:, 1]
            a = self.function.shape_function.nu(first_inspections)
            first_increment_contribution = -np.log(
                gamma.cdf(
                    self.first_increment_uncertainty[1]
                    - self.deterioration_data.values[:, 1],
                    a=a,
                    scale=1 / self.rate,
                )
                - gamma.cdf(
                    self.first_increment_uncertainty[0]
                    - self.deterioration_data.values[:, 1],
                    a=a,
                    scale=1 / self.rate,
                )
            )
            contributions[:, 0] = first_increment_contribution[:, None]


class GammaProcess(ParametricModel):
    """
    BLABLABLABLA
    """

    shape_names: tuple = ("exponential", "power")

    def __init__(
        self,
        shape: str,
        rate: Optional[float] = None,
        **shape_params: Union[float, None],
    ):
        super().__init__()
        # if shape == "exponential":
        #     shape_functions = ExponentialShapeFunctions(**shape_params)
        if shape == "power":
            shape_functions = PowerShape(**shape_params)
        else:
            raise ValueError(
                f"{shape} is not valid name for shape, only {self.shape_names} are allowed"
            )
        self.new_params(rate=rate)
        self.compose_with(
            process_lifetime_distribution=GammaProcessDistribution(shape_functions)
        )

    def init_params(self, *args: Any) -> NDArray[np.float64]:
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
        inspection_times: NDArray[np.float64],
        unit_ids=None,
        nb_sample=1,
        seed=None,
        add_death_time=True,
    ) -> tuple[NDArray[np.float64]]:
        """
        inspection_times has 0 at first
        inspection_times is sorted with unit_ids
        ([0, 2, 3, 4, 5, 10, 0, 3, 6, 11,...], [0, 0, 0, 0, 0, 0, 1, 1 ,1, 1, ...])
        N : nb_units
        M : nb_measures
        B : nb_sample

        inspection_times : (N*M,)
        unit_ids : (N*M,)
        """

        np.random.seed(seed)
        if unit_ids is None:
            unit_ids = np.zeros_like(inspection_times)

        _, counts_of_ids = np.unique(unit_ids, return_counts=True)  # (N,) (N,)

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

    def _init_likelihood(
        self,
        deterioration_data: Deteriorations,
        first_increment_uncertainty,
        measurement_tol,
        **kwargs: Any,
    ) -> LikelihoodFromDeteriorations:
        if len(kwargs) != 0:
            extra_args_names = tuple(kwargs.keys())
            raise ValueError(
                f"""
                Distribution likelihood does not expect other data than lifetimes
                Remove {extra_args_names} from kwargs.
                """
            )
        return LikelihoodFromDeteriorations(
            self.function.copy(),
            deterioration_data,
            first_increment_uncertainty=first_increment_uncertainty,
            measurement_tol=measurement_tol,
        )

    def fit(
        self,
        deterioration_measurements: NDArray[np.float64],
        inspection_times: NDArray[np.float64],
        unit_ids: NDArray[np.float64],
        first_increment_uncertainty: Optional[tuple] = None,
        measurement_tol: np.floating[Any] = np.finfo(float).resolution,
        inplace: bool = True,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """
        BLABLABLABLA
        """

        deterioration_data = deteriorations_factory(
            array_factory(deterioration_measurements),
            array_factory(inspection_times),
            array_factory(unit_ids),
            self.function.process_lifetime_distribution.initial_resistance,
        )

        param0 = kwargs.pop("x0", self.function.init_params())

        minimize_kwargs = {
            "method": kwargs.pop("method", "Nelder-Mead"),
            "bounds": kwargs.pop("bounds", self.function.params_bounds),
            "constraints": kwargs.pop("constraints", ()),
            "tol": kwargs.pop("tol", None),
            "callback": kwargs.pop("callback", None),
            "options": kwargs.pop("options", None),
        }

        likelihood = self._init_likelihood(
            deterioration_data, first_increment_uncertainty, measurement_tol, **kwargs
        )

        optimizer = minimize(
            likelihood.negative_log,
            param0,
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )

        if inplace:
            self.params = optimizer.x
        return optimizer.x
