from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from relife.economic import ExponentialDiscounting, Reward, cost
from relife.lifetime_model import (
    FrozenAgeReplacementModel,
    FrozenLeftTruncatedModel,
    FrozenLifetimeRegression,
    LifetimeDistribution,
)

# from relife.stochastic_process import (
#     RenewalProcessIterator,
#     RenewalRewardProcessIterator,
# )
from relife.stochastic_process import RenewalRewardProcess

M = TypeVar("M", LifetimeDistribution, FrozenLifetimeRegression, FrozenAgeReplacementModel, FrozenLeftTruncatedModel)
R = TypeVar("R", bound=Reward)


class BaseOneCycleAgeReplacementPolicy(Generic[M, R]):

    def __init__(
        self,
        lifetime_model: M,
        reward: R,
        discounting_rate: float = 0.0,
        period_before_discounting: float = 1.0,
    ) -> None:
        self.cost = cost
        self.reward = reward
        self.discounting = ExponentialDiscounting(discounting_rate)
        if period_before_discounting <= 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting
        self.lifetime_model = lifetime_model
        self.count_data = None

    @property
    def discounting_rate(self):
        return self.discounting.rate

    def _make_timeline(self, tf: float, nb_steps: int) -> NDArray[np.float64]:
        # control with reward too
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
        args_nb_assets = getattr(self.lifetime_model, "args_nb_assets", 1)  # default 1 for LifetimeDistribution case
        if args_nb_assets > 1:
            timeline = np.tile(timeline, (args_nb_assets, 1))
        elif self.reward.ndim == 2:  # elif because we consider that if m > 1 in frozen_model, in reward it is 1 or m
            timeline = np.tile(timeline, (self.reward.size, 1))
        return timeline  # (nb_steps,) or (m, nb_steps)

    def expected_total_cost(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = self._make_timeline(tf, nb_steps)
        # reward partial expectation
        return timeline, self.lifetime_model.ls_integrate(
            lambda x: self.reward.conditional_expectation(x) * self.discounting.factor(x), np.zeros_like(timeline), timeline, deg=15
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        # reward partial expectation
        return self.lifetime_model.ls_integrate(
            lambda x: self.reward.conditional_expectation(x) * self.discounting.factor(x), 0.0, np.inf, deg=15
        )  # () or (m, 1)

    def _expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # timeline : (nb_steps,) or (m, nb_steps)
        def f(x: float | NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            # avoid zero division + 1e-6
            return (
                self.reward.conditional_expectation(x)
                * self.discounting.factor(x)
                / (self.discounting.annuity_factor(x) + 1e-6)
            )

        q0 = self.lifetime_model.cdf(self.period_before_discounting) * f(self.period_before_discounting)  # () or (m, 1)
        a = np.full_like(timeline, self.period_before_discounting)  # (nb_steps,) or (m, nb_steps)
        # change first value of lower bound to compute the integral

        # TODO : filtrer par indice où tr == 0 et appliquer np.nan pour eux (les autres OK)

        a[timeline < self.period_before_discounting] = 0.0  # (nb_steps,)
        # a = np.where(timeline < self.period_before_discounting, 0., a)  # (nb_steps,)
        integral = self.lifetime_model.ls_integrate(
            f, a, timeline, deg=15
        )  # (nb_steps,) or (m, nb_steps) if q0: (), or (m, nb_steps) if q0 : (m, 1)
        mask = np.broadcast_to(
            timeline < self.period_before_discounting, integral.shape
        )  # (), (nb_steps,) or (m, nb_steps)
        q0 = np.broadcast_to(q0, integral.shape)  # (nb_steps,) or (m, nb_steps)
        integral = np.where(mask, q0, q0 + integral)
        return timeline, integral

    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = self._make_timeline(tf, nb_steps)  # (nb_steps,) or (m, nb_steps)
        return self._expected_equivalent_annual_cost(timeline)  # (nb_steps,) or (m, nb_steps)

    def asymptotic_expected_equivalent_annual_cost(
        self,
    ) -> NDArray[np.float64]:
        timeline = np.array(np.inf)
        args_nb_assets = getattr(self.lifetime_model, "args_nb_assets", 1)  #  default 1 for LifetimeDistribution case
        if args_nb_assets > 1:
            timeline = np.tile(timeline, (args_nb_assets, 1))
        elif self.reward.ndim == 2:  # elif because we consider that if m > 1 in frozen_model, in reward it is 1 or m
            timeline = np.tile(timeline, (self.reward.size, 1))
        # timeline : () or (m, 1)
        return self._expected_equivalent_annual_cost(timeline)[-1]  # () or (m, 1)

    # def sample(
    #     self,
    #     tf: float,
    #     t0: float = 0.0,
    #     size: int | tuple[int] | tuple[int, int] = 1,
    #     seed: Optional[int] = None,
    # ) -> None:
    #     from relife.stochastic_process.sample import RenewalProcessIterable
    #
    #     iterator = RenewalProcessIterable(self, size, (t0, tf), seed=seed)
    #     self.count_data = concatenate_count_data(islice(iterator, 1), maxsample)
    #
    # def generate_lifetime_data(
    #     self,
    #     tf: float,
    #     t0: float = 0.0,
    #     size: int | tuple[int] | tuple[int, int] = 1,
    #     maxsample: int = 1e5,
    #     seed: Optional[int] = None,
    # ) -> LifetimeData:
    #     from relife.stochastic_process.sample import concatenate_count_data
    #
    #     iterator = RenewalProcessIterator(self, size, (t0, tf), seed=seed)
    #     count_data = concatenate_count_data(islice(iterator, 1), maxsample)
    #     return LifetimeData(
    #         count_data.struct_array["time"].copy(),
    #         event=count_data.struct_array["event"].copy(),
    #         entry=count_data.struct_array["entry"].copy(),
    #         args=tuple(
    #             (
    #                 np.take(arg, count_data.struct_array["asset_id"])
    #                 for arg in getattr(self.lifetime_model, "frozen_args", ())
    #             )
    #         ),
    #     )


class BaseAgeReplacementPolicy(Generic[M, R]):

    def __init__(self, stochastic_process: RenewalRewardProcess[M, R]):
        self.stochastic_process = stochastic_process

    @property
    def discounting_rate(self):
        return self.stochastic_process.discounting_rate

    @discounting_rate.setter
    def discounting_rate(self, value: float) -> None:
        self.stochastic_process.discounting_rate = value

    def expected_nb_replacements(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        The expected number of replacements.

        It is computed  by solving the renewal equation:

        .. math::

            m(t) = F_1(t) + \int_0^t m(t-x) \mathrm{d}F(x)

        where:

        - :math:`m` is the renewal function,
        - :math:`F` is the cumulative distribution function of the underlying
          lifetime model,
        - :math:`F_1` is the cumulative distribution function of the underlying
          lifetime model for the fist renewal in the case of a delayed renewal
          process.


        Parameters
        ----------
        tf : float
            Time horizon. The expected number of replacements will be computed up until this calendar time.
        nb_steps : int
            The number of steps used to compute the expected number of replacements

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected number of replacements and its corresponding values at each
            step of the timeline.
        """
        return self.stochastic_process.renewal_function(tf, nb_steps)  # (nb_steps,) or (m, nb_steps)

    def expected_total_cost(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        The expected total cost.

        It is computed by solving the renewal equation and is given by:

        .. math::

            z(t) = \mathbb{E}(Z_t) = \int_{0}^{\infty}\mathbb{E}(Z_t~|~X_1 = x)dF(x)

        where :

        - :math:`t` is the time
        - :math:`X_i \sim F` are :math:`n` random variable lifetimes, *i.i.d.*, of cumulative distribution :math:`F`.
        - :math:`Z_t` is the random variable reward at each time :math:`t`.
        - :math:`\delta` is the discounting rate.

        Parameters
        ----------
        tf : float
            Time horizon. The expected total cost will be computed up until this calendar time.
        nb_steps : int
            The number of steps used to compute the expected total cost

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected total cost and its corresponding values at each
            step of the timeline.
        """
        return self.stochastic_process.expected_total_reward(tf, nb_steps)  # (nb_steps,) or (m, nb_steps)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        r"""
        The asymtotic expected total cost

        .. math::

            \lim_{t\to\infty} z(t)

        where :math:`z(t)` is the expected total cost at :math:`t`. See :py:meth:`~AgeReplacementPolicy.expected_total_cost` for more details.

        Returns
        -------
        ndarray
            The asymptotic expected total cost values
        """
        return self.stochastic_process.asymptotic_expected_total_reward()  # () or (m, 1)

    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        The expected equivalent annual cost.

        .. math::

            \text{EEAC}(t) = \dfrac{\delta z(t)}{1 - e^{-\delta t}}

        where :

        - :math:`t` is the time
        - :math:`z(t)` is the expected_total_cost at :math:`t`. See :py:meth:`~AgeReplacementPolicy.expected_total_cost` for more details.`.
        - :math:`\delta` is the discounting rate.

        Parameters
        ----------
        tf : float
            Time horizon. The expected equivalent annual cost will be computed up until this calendar time.
        nb_steps : int
            The number of steps used to compute the expected equivalent annual cost

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected annual cost and its corresponding values at each
            step of the timeline.
        """
        return self.stochastic_process.expected_equivalent_annual_worth(tf, nb_steps)  # (nb_steps,) or (m, nb_steps)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        r"""
        The asymtotic expected equivalent annual cost

        .. math::

            \lim_{t\to\infty} \text{EEAC}(t)

        where :math:`\text{EEAC}(t)` is the expected equivalent annual cost at :math:`t`. See :py:meth:`~AgeReplacementPolicy.expected_equivalent_annual_cost` for more details.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost
        """
        return self.stochastic_process.asymptotic_expected_equivalent_annual_worth()  # () or (m, 1)

    # def sample_count_data(
    #     self,
    #     tf: float,
    #     t0: float = 0.0,
    #     size: int | tuple[int] | tuple[int, int] = 1,
    #     maxsample: int = 1e5,
    #     seed: Optional[int] = None,
    # ) -> None:
    #     self.stochastic_process.sample_count_data(tf, t0, size, maxsample, seed)
    #
    # def generate_lifetime_data(
    #     self,
    #     tf: float,
    #     t0: float = 0.0,
    #     size: int | tuple[int] | tuple[int, int] = 1,
    #     maxsample: int = 1e5,
    #     seed: Optional[int] = None,
    # ) -> LifetimeData:
    #     return self.stochastic_process.sample_count_data(tf, t0, size, maxsample, seed)


# @overload
# def age_replacement_policy(
#     lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
#     cost: Cost,
#     one_cycle = True,
#     a0: Optional[float | NDArray[np.float64]] = None,
#     ar: Optional[float | NDArray[np.float64]] = None,
#     ar1 = None,
#     discounting_rate: float = 0.0,
# ) -> OneCycleAgeReplacementPolicy: ...
#
#
# @overload
# def age_replacement_policy(
#     lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
#     cost: Cost,
#     one_cycle = False,
#     a0: Optional[float | NDArray[np.float64]] = None,
#     ar: Optional[float | NDArray[np.float64]] = None,
#     ar1: Optional[float | NDArray[np.float64]] = None,
#     discounting_rate: float = 0.0,
# ) -> AgeReplacementPolicy: ...
#
#
# def age_replacement_policy(
#     lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
#     cost: Cost,
#     one_cycle: bool = False,
#     a0: float | NDArray[np.float64] = None,
#     ar: Optional[float | NDArray[np.float64]] = None,
#     ar1: Optional[float | NDArray[np.float64]] = None,
#     discounting_rate: float = 0.0,
# ) -> OneCycleAgeRenewalPolicy | AgeReplacementPolicy:
#
#     ar = np.nan if ar is None else ar
#     if not one_cycle:
#         first_lifetime_model = None
#         if a0 is not None and ar1 is not None:
#             first_lifetime_model = freeze(AgeReplacementModel(LeftTruncatedModel(lifetime_model)), ar1, a0)
#         elif ar1 is not None:
#             first_lifetime_model = freeze(AgeReplacementModel(lifetime_model), ar1)
#         elif a0 is not None:
#             first_lifetime_model = freeze(AgeReplacementModel(LeftTruncatedModel(lifetime_model)), ar, a0)
#         return AgeReplacementPolicy(
#             freeze(AgeReplacementModel(lifetime_model), ar),
#             cost,
#             ExponentialDiscounting(discounting_rate),
#             first_lifetime_model=first_lifetime_model
#         )
#     else:
#         if ar1 is not None:
#             raise ValueError
#         if a0 is not None and
#
#
#
#     return DefaultAgeReplacementPolicy(
#         model,
#         cost["cf"],
#         cost["cp"],
#         discounting_rate=discounting_rate,
#         ar=ar,
#         ar1=ar1,
#         a0=a0,
#         model1=model1,
#     )
#
