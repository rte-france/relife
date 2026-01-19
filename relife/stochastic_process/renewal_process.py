# pyright: basic

import copy
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife.base import ParametricModel
from relife.economic import ExponentialDiscounting, Reward
from relife.lifetime_model import (
    LeftTruncatedModel,
)
from relife.typing import AnyParametricLifetimeModel, Seed
from relife.utils import is_frozen

from ._renewal_equations import (
    delayed_renewal_equation_solver,
    renewal_equation_solver,
)
from ._sample import RenewalProcessSample, RenewalRewardProcessSample


def _make_timeline(tf: float, nb_steps: int) -> NDArray[np.float64]:
    timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
    return np.atleast_2d(timeline)  # (1, nb_steps) to ensure broadcasting


class LifetimeFitArgs(TypedDict):
    time: NDArray[np.floating]
    event: NDArray[np.bool_]
    entry: NDArray[np.floating]
    args: tuple[NDArray[np.floating], ...]


class RenewalProcess(ParametricModel):
    # noinspection PyUnresolvedReferences
    """Renewal process.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.

    first_lifetime_model : any lifetime distribution or frozen lifetime model, optional
        A lifetime model for the first renewal (delayed renewal process). It is None by default

    Attributes
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.

    first_lifetime_model : any lifetime distribution or frozen lifetime model, optional
        A lifetime model for the first renewal (delayed renewal process). It is None by default
    nb_params
    params
    params_names
    """

    lifetime_model: AnyParametricLifetimeModel[()]
    first_lifetime_model: AnyParametricLifetimeModel[()] | None

    def __init__(
        self,
        lifetime_model: AnyParametricLifetimeModel[()],
        first_lifetime_model: AnyParametricLifetimeModel[()] | None = None,
    ) -> None:
        super().__init__()
        self.lifetime_model = lifetime_model
        self.first_lifetime_model = first_lifetime_model

    def renewal_function(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""The renewal function.

        The renewal function gives the expected total number of renewals. It is computed  by solving the
        renewal equation:

        .. math::

            m(t) = F_1(t) + \int_0^t m(t-x) \mathrm{d}F(x)

        where:

        - :math:`m` is the renewal function.
        - :math:`F` is the cumulative distribution function of the underlying
          lifetime model.
        - :math:`F_1` is the cumulative distribution function of the underlying
          lifetime model for the fist renewal in the case of a delayed renewal
          process.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretized the time.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.

        References
        ----------
        .. [1] Rausand, M., Barros, A., & Hoyland, A. (2020). System Reliability
            Theory: Models, Statistical Methods, and Applications. John Wiley &
            Sons.
        """

        timeline = _make_timeline(tf, nb_steps)  # (nb_steps,) or (1, nb_steps)
        renewal_function = renewal_equation_solver(
            timeline,
            self.lifetime_model,
            self.lifetime_model.cdf if self.first_lifetime_model is None else self.first_lifetime_model.cdf,
        )
        return np.squeeze(timeline), np.squeeze(renewal_function)

    def renewal_density(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""The renewal density.

        The renewal density corresponds to the derivative of the renewal function with
        respect to time. It is computed by solving the renewal equation:

        .. math::

            \mu(t) = f_1(t) + \int_0^t \mu(t-x) \mathrm{d}F(x)

        where:

        - :math:`\mu` is the renewal function.
        - :math:`F` is the cumulative distribution function of the underlying
          lifetime model.
        - :math:`f_1` is the probability density function of the underlying
          lifetime model for the fist renewal in the case of a delayed renewal
          process.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretized the time.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.

        References
        ----------
        .. [1] Rausand, M., Barros, A., & Hoyland, A. (2020). System Reliability
            Theory: Models, Statistical Methods, and Applications. John Wiley &
            Sons.
        """
        timeline = _make_timeline(tf, nb_steps)  #  (nb_steps,) or (m, nb_steps)
        renewal_density = renewal_equation_solver(
            timeline,
            self.lifetime_model,
            self.lifetime_model.pdf if self.first_lifetime_model is None else self.first_lifetime_model.pdf,
        )
        return np.squeeze(timeline), np.squeeze(renewal_density)

    def sample(self, size: int, tf: float, t0: float = 0.0, seed: Seed | None = None) -> RenewalProcessSample:
        """Renewal data sampling.

        This function will sample data and encapsulate them in an object.

        Parameters
        ----------
        size : int
            The size of the desired sample
        tf : float
            Time at the end of the observation.
        t0 : float, default 0
            Time at the beginning of the observation.
        size : int or tuple of 2 int
            Size of the sample
        seed : int, optional
            Random seed, by default None.

        """

        from ._sample import RenewalProcessIterable

        iterable = RenewalProcessIterable(self, size, tf, t0=t0, seed=seed)
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("sample_id", "asset_id", "timeline"))
        return RenewalProcessSample(t0, tf, struct_array)

    def generate_failure_data(self, size: int, tf: float, t0: float = 0.0, seed: Seed | None = None) -> LifetimeFitArgs:
        """Generate lifetime data

        This function will generate lifetime data that can be used to fit a lifetime model.

        Parameters
        ----------
        size : int
            The size of the desired sample
        tf : float
            Time at the end of the observation.
        t0 : float, default 0
            Time at the beginning of the observation.
        nb_assets : int, optional
            Number of assets.
        seed : int, optional
            Random seed, by default None.

        Returns
        -------
        A dict of time, event, entry and args (covariates)

        """
        from ._sample import RenewalProcessIterable

        if self.first_lifetime_model is not None and self.first_lifetime_model != self.lifetime_model:
            if is_frozen(self.first_lifetime_model):
                if (
                    isinstance(self.first_lifetime_model.unfrozen_model, LeftTruncatedModel)
                    and self.first_lifetime_model.unfrozen_model.baseline == self.lifetime_model
                ):
                    pass
            else:
                raise ValueError(
                    f"Calling sample_lifetime_data with lifetime_model different from first_lifetime_model is ambiguous."
                )
        iterable = RenewalProcessIterable(self, size, tf, t0=t0, nb_assets=nb_assets, seed=seed)
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("sample_id", "asset_id", "timeline"))

        nb_assets = int(np.max(struct_array["asset_id"])) + 1
        args_2d = tuple((np.atleast_2d(arg) for arg in getattr(self.lifetime_model, "args", ())))
        # broadcasted_args = tuple((np.broadcast_to(arg, (nb_assets, arg.shape[-1])) for arg in args_2d))
        tuple_args_arr = tuple((np.take(np.asarray(arg), struct_array["asset_id"], axis=0) for arg in args_2d))

        returned_dict = {
            "time": struct_array["time"].copy(),
            "event": struct_array["event"].copy(),
            "entry": struct_array["entry"].copy(),
            "args": tuple_args_arr,
        }
        return returned_dict


class RenewalRewardProcess(RenewalProcess):
    # noinspection PyUnresolvedReferences
    """Renewal reward process.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.
    reward : Reward
        A reward object that answers costs or conditional costs given lifetime values
    discounting_rate : float
        The discounting rate value used in the exponential discounting function
    first_lifetime_model : any lifetime distribution or frozen lifetime model, optional
        A lifetime model for the first renewal (delayed renewal process). It is None by default
    reward : Reward
        A reward object for the first renewal

    Attributes
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.
    first_lifetime_model : any lifetime distribution or frozen lifetime model, optional
        A lifetime model for the first renewal (delayed renewal process). It is None by default
    reward : Reward
        A reward object that answers costs or conditional costs given lifetime values
    first_reward : Reward
        A reward object for the first renewal. If it is not given at the initialization, it is a copy of reward.
    discounting_rate
    nb_params
    params
    params_names
    """
    lifetime_model: AnyParametricLifetimeModel[()]
    first_lifetime_model: AnyParametricLifetimeModel[()] | None
    reward: Reward
    first_reward: Reward
    discounting: ExponentialDiscounting

    def __init__(
        self,
        lifetime_model: AnyParametricLifetimeModel[()],
        reward: Reward,
        discounting_rate: float = 0.0,
        first_lifetime_model: AnyParametricLifetimeModel[()] | None = None,
        first_reward: Reward | None = None,
    ) -> None:
        super().__init__(lifetime_model, first_lifetime_model)
        self.reward = reward
        self.first_reward = first_reward if first_reward is not None else copy.deepcopy(reward)
        self.discounting = ExponentialDiscounting(discounting_rate)

    @property
    def discounting_rate(self) -> float:
        """
        The discounting rate value
        """
        return self.discounting.rate

    @discounting_rate.setter
    def discounting_rate(self, value: float) -> None:
        self.discounting.rate = value

    def expected_total_reward(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""The expected total reward.

        The renewal equation solved to compute the expected reward is:

        .. math::

            z(t) = \int_0^t E[Y | X = x] e^{-\delta x} \mathrm{d}F(x) + \int_0^t z(t-x)
            e^{-\delta x}\mathrm{d}F(x)

        where:

        - :math:`z` is the expected total reward.
        - :math:`F` is the cumulative distribution function of the underlying
          lifetime model.
        - :math:`X` the interarrival random variable.
        - :math:`Y` the associated reward.
        - :math:`D` the exponential discount factor.

        If the renewal reward process is delayed, the expected total reward is
        modified as:

        .. math::

            z_1(t) = \int_0^t E[Y_1 | X_1 = x] e^{-\delta x} \mathrm{d}F_1(x) + \int_0^t
            z(t-x) e^{-\delta x} \mathrm{d}F_1(x)

        where:

        - :math:`z_1` is the expected total reward with delay.
        - :math:`F_1` is the cumulative distribution function of the lifetime
          model for the first renewal.
        - :math:`X_1` the interarrival random variable of the first renewal.
        - :math:`Y_1` the associated reward of the first renewal.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretized the time.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.

        """
        timeline = _make_timeline(tf, nb_steps)  #  (nb_steps,) or (m, nb_steps)
        z = renewal_equation_solver(
            timeline,
            self.lifetime_model,
            lambda t: self.lifetime_model.ls_integrate(
                lambda x: self.reward.conditional_expectation(x) * self.discounting.factor(x),
                np.zeros_like(t),
                np.asarray(t),
                deg=15,
            ),  # reward partial expectation
            discounting=self.discounting,
        )
        if self.first_lifetime_model is not None:
            z = delayed_renewal_equation_solver(
                timeline,
                z,
                self.first_lifetime_model,
                lambda t: self.first_lifetime_model.ls_integrate(
                    lambda x: self.first_reward.conditional_expectation(x) * self.discounting.factor(x),
                    np.zeros_like(t),
                    np.asarray(t),
                    deg=15,
                ),  # reward partial expectation
                discounting=self.discounting,
            )
        return np.squeeze(timeline), np.squeeze(z)  # (nb_steps,), (nb_steps,) or (m, nb_steps)

    def asymptotic_expected_total_reward(self) -> np.float64 | NDArray[np.float64]:
        r"""Asymptotic expected total reward.

        The asymptotic expected total reward is:

        .. math::

            z^\infty = \lim_{t\to \infty} z(t) = \dfrac{E\left[Y e^{-\delta X}\right]}{1-E\left[e^{-\delta X}\right]}

        where:

        - :math:`X` the interarrival random variable.
        - :math:`Y` the associated reward.
        - :math:`D` the exponential discount factor.

        If the renewal reward process is delayed, the asymptotic expected total
        reward is modified as:

        .. math::

            z_1^\infty = E\left[Y_1 e^{-\delta X_1}\right] + z^\infty E\left[e^{-\delta X_1}\right]

        where:

        - :math:`X_1` the interarrival random variable of the first renewal.
        - :math:`Y_1` the associated reward of the first renewal.

        Returns
        -------
        ndarray
            The assymptotic expected total reward of the process.
        """
        lf = self.lifetime_model.ls_integrate(
            lambda x: self.discounting.factor(x), np.float64(0.0), np.asarray(np.inf), deg=100
        )  # () or (m, 1)
        if self.discounting_rate == 0.0:
            return np.full_like(np.squeeze(lf), np.inf)
        ly = self.lifetime_model.ls_integrate(
            lambda x: self.discounting.factor(x) * self.reward.conditional_expectation(x), 0.0, np.inf, deg=100
        )  # () or (m, 1)
        z = np.squeeze(ly / (1 - lf))  # () or (m,)
        if self.first_lifetime_model is not None:
            lf1 = np.squeeze(
                self.first_lifetime_model.ls_integrate(lambda x: self.discounting.factor(x), 0.0, np.inf, deg=100)
            )  # () or (m,)
            ly1 = np.squeeze(
                self.first_lifetime_model.ls_integrate(
                    lambda x: self.discounting.factor(x) * self.first_reward.conditional_expectation(x),
                    0.0,
                    np.inf,
                    deg=100,
                )
            )  # () or (m,)
            z = ly1 + z * lf1  # () or (m,)
        return z  # () or (m,)

    def expected_equivalent_annual_worth(
        self, tf: float, nb_steps: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Expected equivalent annual worth.

        Gives the equivalent annual worth of the expected total reward of the
        process at each point of the timeline.

        The equivalent annual worth at time :math:`t` is equal to the expected
        total reward :math:`z` divided by the annuity factor :math:`AF(t)`.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretized the time.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.
        """
        timeline, z = self.expected_total_reward(
            tf, nb_steps
        )  # timeline : (nb_steps,) z : (nb_steps,) or (m, nb_steps)
        af = self.discounting.annuity_factor(timeline)  # (nb_steps,)
        if z.ndim == 2 and af.shape != z.shape:  # (m, nb_steps)
            af = np.tile(af, (z.shape[0], 1))  # (m, nb_steps)
        q = z / (af + 1e-6)  # # (nb_steps,) or (m, nb_steps) avoid zero division
        if self.first_lifetime_model is not None:
            q0 = self.first_reward.conditional_expectation(np.asarray(0.0)) * self.first_lifetime_model.pdf(0.0)
        else:
            q0 = self.reward.conditional_expectation(np.asarray(0.0)) * self.lifetime_model.pdf(0.0)
        # q0 : () or (m, 1)
        q0 = np.broadcast_to(q0, af.shape)  # (), (nb_steps,) or (m, nb_steps)
        eeac = np.where(af == 0, q0, q)  # (nb_steps,) or (m, nb_steps)
        return np.squeeze(timeline), np.squeeze(eeac)  # (nb_steps,) and (nb_steps) or (m, nb_steps)

    def asymptotic_expected_equivalent_annual_worth(self) -> np.float64 | NDArray[np.float64]:
        """Asymptotic expected equivalent annual worth.

        Returns
        -------
        ndarray
            The assymptotic expected equivalent annual worth.
        """
        if self.discounting_rate == 0.0:
            return np.squeeze(
                np.asarray(
                    self.lifetime_model.ls_integrate(
                        lambda x: self.reward.conditional_expectation(x), np.float64(0.0), np.asarray(np.inf), deg=100
                    ),
                    dtype=float,
                )
                / self.lifetime_model.mean()
            )  # () or (m,)
        return self.discounting_rate * self.asymptotic_expected_total_reward()  # () or (m,)

    @override
    def sample(self, size: int, tf: float, t0: float = 0.0, seed: Seed | None = None) -> RenewalRewardProcessSample:
        from ._sample import RenewalProcessIterable

        iterable = RenewalProcessIterable(self, size, tf, t0=t0, seed=seed)
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("nb_renewal", "asset_id", "sample_id"))
        return RenewalRewardProcessSample(t0, tf, struct_array)
