import copy
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
from optype.numpy import Array, Array1D, Array2D, ArrayND, is_array_1d

from relife.base import ParametricModel
from relife.lifetime_models._base import (
    ParametricLifetimeModel,
)
from relife.lifetime_models._conditional_models import (
    get_conditional_lifetime_model,
)
from relife.rewards import ExponentialDiscounting, Reward
from relife.stochastic_processes._sample import StochasticSampleMapping
from relife.utils import to_column_2d_if_1d

from ._renewal_equations import (
    delayed_renewal_equation_solver,
    renewal_equation_solver,
)

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


__all__ = [
    "RenewalProcess",
    "RenewalRewardProcess",
    "make_timeline",
]


def make_timeline(
    tf: float, nb_steps: int
) -> Array[tuple[Literal[1], int], np.float64]:
    return np.atleast_2d(np.linspace(0, tf, nb_steps, dtype=np.float64))


def reshape_a0_ar(func):
    def wrapper(*args, **kwargs):

        if kwargs.get("a0"):
            if kwargs.get("a0") != 0.0:
                kwargs["a0"] = to_column_2d_if_1d(kwargs["a0"])

        if kwargs.get("ar"):
            if kwargs.get("ar") != np.inf:
                kwargs["ar"] = to_column_2d_if_1d(kwargs["ar"])

        return func(*args, **kwargs)

    return wrapper


class LifetimeFitArgs(TypedDict):
    time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64]
    event: Array1D[np.bool_]
    entry: Array1D[np.float64]
    args: Array1D[Any] | Array2D[Any] | tuple[Array1D[Any] | Array2D[Any], ...]


class RenewalProcess(ParametricModel):
    """Renewal process.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.

    first_lifetime_model : any lifetime distribution or frozen lifetime model, optional
        A lifetime model for the first renewal (delayed renewal process). It is lifetime_model by default.

    Attributes
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.

    first_lifetime_model : any lifetime distribution or frozen lifetime model, optional
        A lifetime model for the first renewal (delayed renewal process). It is lifetime_model by default
    nb_params
    params
    params_names
    """  # noqa: E501

    lifetime_model: ParametricLifetimeModel[()]
    first_lifetime_model: ParametricLifetimeModel[()]

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        first_lifetime_model: ParametricLifetimeModel[()] | None = None,
    ) -> None:
        super().__init__()
        self.lifetime_model = lifetime_model
        if first_lifetime_model is None:
            first_lifetime_model = lifetime_model
        self.first_lifetime_model = first_lifetime_model

    @reshape_a0_ar
    def renewal_function(
        self,
        tf: float,
        nb_steps: int,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        r"""The renewal function.

        The renewal function gives the expected total number of renewals.
        It is computed  by solving the renewal equation:

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
        a0 : float or np.ndarray or None
            Initial ages
        ar : float or np.ndarray or None
            Preventive ages of replacements

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

        timeline = make_timeline(tf, nb_steps)
        renewal_function = renewal_equation_solver(
            timeline,
            get_conditional_lifetime_model(self.lifetime_model, ar=ar),
            get_conditional_lifetime_model(self.first_lifetime_model, ar=ar, a0=a0).cdf,
        )
        return np.squeeze(timeline), np.squeeze(renewal_function)

    @reshape_a0_ar
    def expected_number_of_events(
        self,
        tf: float,
        nb_steps: int,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        r"""The expected number of events (no preventive renewals) at each time of the process.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretized the time.
        a0 : float or np.ndarray or None
            Initial ages
        ar : float or np.ndarray or None
            Preventive ages of replacements

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.
        """  # noqa: E501

        timeline = make_timeline(tf, nb_steps)
        lifetime_model = get_conditional_lifetime_model(self.lifetime_model, ar=ar)
        # Apply delay for the first renewal with a0
        # If no a0 are given, will result in the same solution
        first_lifetime_model = get_conditional_lifetime_model(
            self.first_lifetime_model, a0=a0, ar=ar
        )

        def F(t: ST | NumpyST | ArrayND[NumpyST]) -> np.float64 | ArrayND[np.float64]:
            return self.lifetime_model.cdf(np.minimum(t, ar))

        def F1(t: ST | NumpyST | ArrayND[NumpyST]) -> np.float64 | ArrayND[np.float64]:
            left_truncated_model = get_conditional_lifetime_model(
                self.first_lifetime_model, a0=a0
            )
            return left_truncated_model.cdf(np.minimum(t, ar - a0))

        z = renewal_equation_solver(timeline, lifetime_model, F)
        z = delayed_renewal_equation_solver(timeline, z, first_lifetime_model, F1)
        return np.squeeze(timeline), np.squeeze(z)

    @reshape_a0_ar
    def expected_number_of_preventive_renewals(
        self,
        tf: float,
        nb_steps: int,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        r"""The expected number of preventive renewals (no events) at each time of the process.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretized the time.
        a0 : float or np.ndarray or None
            Initial ages
        ar : float or np.ndarray or None
            Preventive ages of replacements

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.
        """  # noqa: E501

        timeline = make_timeline(tf, nb_steps)

        def F(t: ST | NumpyST | ArrayND[NumpyST]) -> np.float64 | ArrayND[np.float64]:
            return (1 - self.lifetime_model.cdf(ar)) * (t > ar)

        def F1(t: ST | NumpyST | ArrayND[NumpyST]) -> np.float64 | ArrayND[np.float64]:
            first_ar = ar - a0
            return (
                1
                - get_conditional_lifetime_model(self.first_lifetime_model, a0=a0).cdf(
                    first_ar
                )
            ) * (t > first_ar)

        z = renewal_equation_solver(
            timeline, get_conditional_lifetime_model(self.lifetime_model, ar=ar), F
        )

        # Apply delay for the first renewal with a0
        # If no a0 are given, will result in the same solution
        z = delayed_renewal_equation_solver(
            timeline,
            z,
            get_conditional_lifetime_model(self.first_lifetime_model, a0=a0, ar=ar),
            F1,
        )

        return np.squeeze(timeline), np.squeeze(z)

    @reshape_a0_ar
    def renewal_density(
        self,
        tf: float,
        nb_steps: int,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
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
        a0 : float or np.ndarray or None
            Initial ages
        ar : float or np.ndarray or None
            Preventive ages of replacements

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
        timeline = make_timeline(tf, nb_steps)
        renewal_density = renewal_equation_solver(
            timeline,
            get_conditional_lifetime_model(self.lifetime_model, ar=ar),
            get_conditional_lifetime_model(self.first_lifetime_model, ar=ar, a0=a0).pdf,
        )
        return np.squeeze(timeline), np.squeeze(renewal_density)

    @reshape_a0_ar
    def sample(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> StochasticSampleMapping:
        """Renewal data sampling.

        This function will sample data and encapsulate them in an object.

        Parameters
        ----------
        nb_samples : int
            The size of the desired sample
        time_window : tuple of two floats
            Time window in which data are sampled
        a0 : float or np.ndarray or None
            Initial ages
        ar : float or np.ndarray or None
            Preventive ages of replacements
        seed : int, optional
            Random seed, by default None.

        """

        from ._sample import RenewalProcessIterable

        iterable = RenewalProcessIterable(
            self, nb_samples, time_window, a0=a0, ar=ar, seed=seed
        )
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(
            struct_array, order=("asset_id", "sample_id", "timeline")
        )
        return StochasticSampleMapping.from_struct_array(
            struct_array, iterable.nb_assets, nb_samples
        )

    @reshape_a0_ar
    def generate_failure_data(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> LifetimeFitArgs:
        """Generate lifetime data

        This function will generate lifetime data that can be used to fit a lifetime model.

        Parameters
        ----------
        nb_samples : int
            The size of the desired sample
        time_window : tuple of two floats
            Time window in which data are sampled
        a0 : float or np.ndarray or None
            Initial ages
        ar : float or np.ndarray or None
            Preventive ages of replacements
        seed : int, optional
            Random seed, by default None.

        Returns
        -------
        A dict of time, event, entry and args (covariates)

        """

        from ._sample import RenewalProcessIterable

        if self.first_lifetime_model != self.lifetime_model:
            raise ValueError(
                "Calling sample_lifetime_data with lifetime_model different from first_lifetime_model is ambiguous."  # noqa: E501
            )
        iterable = RenewalProcessIterable(
            self, nb_samples, time_window, a0=a0, ar=ar, seed=seed
        )
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(
            struct_array, order=("sample_id", "asset_id", "timeline")
        )

        args_2d = tuple(
            np.atleast_2d(arg) for arg in getattr(self.lifetime_model, "args", ())
        )
        tuple_args_arr = tuple(
            np.take(np.asarray(arg), struct_array["asset_id"], axis=0)
            for arg in args_2d
        )

        return LifetimeFitArgs(
            time=struct_array["time"].copy(),
            event=struct_array["event"].copy(),
            entry=struct_array["entry"].copy(),
            args=tuple_args_arr,
        )


class RenewalRewardProcess(RenewalProcess):
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
        A lifetime model for the first renewal (delayed renewal process). It is lifetime_model by default
    reward : Reward
        A reward object for the first renewal

    Attributes
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.
    first_lifetime_model : any lifetime distribution or frozen lifetime model, optional
        A lifetime model for the first renewal (delayed renewal process). It is lifetime_model by default
    reward : Reward
        A reward object that answers costs or conditional costs given lifetime values
    first_reward : Reward
        A reward object for the first renewal. If it is not given at the initialization, it is a copy of reward.
    discounting_rate
    nb_params
    params
    params_names
    """  # noqa: E501

    reward: Reward
    first_reward: Reward
    discounting: ExponentialDiscounting

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        reward: Reward,
        discounting_rate: float = 0.0,
        first_lifetime_model: ParametricLifetimeModel[()] | None = None,
        first_reward: Reward | None = None,
    ) -> None:
        super().__init__(lifetime_model, first_lifetime_model)
        self.reward = reward
        self.first_reward = (
            first_reward if first_reward is not None else copy.deepcopy(reward)
        )
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

    @reshape_a0_ar
    def expected_total_reward(
        self,
        tf: float,
        nb_steps: int,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
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
        a0 : float or np.ndarray or None
            Initial ages
        ar : float or np.ndarray or None
            Preventive ages of replacements

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.

        """
        timeline = make_timeline(tf, nb_steps)

        def F(t: ST | NumpyST | ArrayND[NumpyST]) -> np.float64 | ArrayND[np.float64]:
            return get_conditional_lifetime_model(
                self.lifetime_model, ar=ar
            ).ls_integrate(
                lambda x: (
                    self.reward.conditional_expectation(x) * self.discounting.factor(x)
                ),
                np.zeros_like(t),
                np.asarray(t),
                deg=15,
            )

        def F1(t: ST | NumpyST | ArrayND[NumpyST]) -> np.float64 | ArrayND[np.float64]:
            return get_conditional_lifetime_model(
                self.first_lifetime_model, a0=a0, ar=ar
            ).ls_integrate(
                lambda x: (
                    self.first_reward.conditional_expectation(x, a0=a0)
                    * self.discounting.factor(x)
                ),
                np.zeros_like(t),
                np.asarray(t),
                deg=15,
            )

        z = renewal_equation_solver(
            timeline,
            get_conditional_lifetime_model(self.lifetime_model, ar=ar),
            F,
            discounting=self.discounting,
        )

        # Apply delay for the first renewal with a0
        # If no a0 are given, will result in the same solution
        z = delayed_renewal_equation_solver(
            timeline,
            z,
            get_conditional_lifetime_model(self.first_lifetime_model, a0=a0, ar=ar),
            F1,
            discounting=self.discounting,
        )
        return np.squeeze(timeline), np.squeeze(z)

    @reshape_a0_ar
    def asymptotic_expected_total_reward(
        self,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> np.float64 | Array1D[np.float64]:
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

        Parameters
        ----------
        a0 : float or np.ndarray or None
            Initial ages
        ar : float or np.ndarray or None
            Preventive ages of replacements

        Returns
        -------
        ndarray
            The assymptotic expected total reward of the process.
        """  # noqa: E501

        lf = get_conditional_lifetime_model(self.lifetime_model, ar=ar).ls_integrate(
            lambda x: self.discounting.factor(x),
            np.float64(0.0),
            np.asarray(np.inf),
            deg=100,
        )  # () or (m, 1)
        if self.discounting_rate == 0.0:
            return np.full_like(np.squeeze(lf), np.inf)
        ly = get_conditional_lifetime_model(self.lifetime_model, ar=ar).ls_integrate(
            lambda x: (
                self.discounting.factor(x) * self.reward.conditional_expectation(x)
            ),
            0.0,
            np.inf,
            deg=100,
        )  # () or (m, 1)
        z = np.squeeze(ly / (1 - lf))  # () or (m,)

        # Apply delay for the first renewal with a0
        # If no a0 are given, will result in the same solution
        lf1 = np.squeeze(
            get_conditional_lifetime_model(
                self.first_lifetime_model, a0=a0, ar=ar
            ).ls_integrate(lambda x: self.discounting.factor(x), 0.0, np.inf, deg=100)
        )  # () or (m,)
        ly1 = np.squeeze(
            get_conditional_lifetime_model(
                self.first_lifetime_model, a0=a0, ar=ar
            ).ls_integrate(
                lambda x: (
                    self.discounting.factor(x)
                    * self.first_reward.conditional_expectation(x, a0)
                ),
                0.0,
                np.inf,
                deg=100,
            )
        )  # () or (m,)
        z = ly1 + z * lf1
        return z

    @reshape_a0_ar
    def expected_equivalent_annual_worth(
        self,
        tf: float,
        nb_steps: int,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
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
        a0 : float or np.ndarray or None
            Initial ages
        ar : float or np.ndarray or None
            Preventive ages of replacements

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.
        """
        timeline, z = self.expected_total_reward(tf, nb_steps, a0=a0, ar=ar)
        af = self.discounting.annuity_factor(timeline)  # (nb_steps,)
        if z.ndim == 2 and af.shape != z.shape:  # (m, nb_steps)
            af = np.tile(af, (z.shape[0], 1))  # (m, nb_steps)
        q = z / (af + 1e-6)  # # (nb_steps,) or (m, nb_steps) avoid zero division
        q0 = self.reward.conditional_expectation(
            np.asarray(0.0)
        ) * get_conditional_lifetime_model(self.lifetime_model, a0=a0).pdf(0.0)
        # q0 : () or (m, 1)
        q0 = np.broadcast_to(q0, af.shape)  # (), (nb_steps,) or (m, nb_steps)
        eeac = np.where(af == 0, q0, q)  # (nb_steps,) or (m, nb_steps)
        return np.squeeze(timeline), np.squeeze(eeac)

    @reshape_a0_ar
    def asymptotic_expected_equivalent_annual_worth(
        self,
        a0: ST | NumpyST | Array1D[NumpyST] = 0.0,
        ar: ST | NumpyST | Array1D[NumpyST] = np.inf,
    ) -> np.float64 | Array1D[np.float64]:
        """Asymptotic expected equivalent annual worth.

        Parameters
        ----------
        a0 : float or np.ndarray or None
            Initial ages
        ar : float or np.ndarray or None
            Preventive ages of replacements

        Returns
        -------
        ndarray
            The assymptotic expected equivalent annual worth.
        """
        if self.discounting_rate == 0.0:
            lifetime_model_applied = get_conditional_lifetime_model(
                self.lifetime_model, ar=ar
            )
            return np.squeeze(
                np.asarray(
                    lifetime_model_applied.ls_integrate(
                        lambda x: self.reward.conditional_expectation(x),
                        np.float64(0.0),
                        np.asarray(np.inf),
                        deg=100,
                    ),
                    dtype=float,
                )
                / lifetime_model_applied.mean()
            )

        res = self.discounting_rate * self.asymptotic_expected_total_reward(a0, ar)
        assert is_array_1d(res) or isinstance(res, np.float64)  # typeguard
        return res
