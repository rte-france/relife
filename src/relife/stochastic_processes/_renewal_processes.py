import functools
import inspect
from collections.abc import Callable
from typing import Any, Literal, ParamSpec, TypeAlias, TypedDict, TypeVar

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

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


__all__ = [
    "RenewalProcess",
    "RenewalRewardProcess",
]

FT: TypeAlias = Callable[
    [ST | NumpyST | ArrayND[NumpyST]],
    np.float64 | ArrayND[np.float64],
]


class RenewalEquationSolver:
    """
    Renewal equation solver.
    """

    lifetime_model: ParametricLifetimeModel[()]
    first_lifetime_model: ParametricLifetimeModel[()] | None
    func: FT
    func1: FT | None

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        func: FT,
        first_lifetime_model: ParametricLifetimeModel[()] | None = None,
        func1: FT | None = None,
    ) -> None:
        self.lifetime_model = lifetime_model
        self.func = func
        if first_lifetime_model:
            assert func1 is not None
        self.first_lifetime_model = first_lifetime_model
        self.func1 = func1

    def solve(
        self, tf: float, nb_steps: int, discounting_rate: float = 0.0
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:

        discounting = ExponentialDiscounting(discounting_rate)
        timeline = np.atleast_2d(np.linspace(0, tf, nb_steps, dtype=np.float64))
        tm = 0.5 * (timeline[:, 1:] + timeline[:, :-1])  # (1, nb_steps - 1)
        f = np.atleast_2d(self.lifetime_model.cdf(timeline))  # (m, nb_steps)
        fm = np.atleast_2d(self.lifetime_model.cdf(tm))  # (m, nb_steps - 1)
        y = np.atleast_2d(self.func(timeline))  # (1, nb_steps)
        d = np.asarray(discounting.factor(timeline))  # (m, nb_steps)
        z = np.empty(y.shape)
        u = d * np.insert(f[:, 1:] - fm, 0, 1, axis=-1)
        v = d[:, :-1] * np.insert(np.diff(fm), 0, 1, axis=-1)
        q0 = 1 / (1 - d[:, 0] * fm[:, 0])
        z[:, 0] = y[:, 0]
        z[:, 1] = q0 * (y[:, 1] + z[:, 0] * u[:, 1])
        for n in range(2, f.shape[-1]):
            z[:, n] = q0 * (
                y[:, n]
                + z[:, 0] * u[:, n]
                + np.sum(z[:, 1:n][:, ::-1] * v[:, 1:n], axis=-1)
            )

        if self.first_lifetime_model is not None and self.func1 is not None:
            f1 = np.atleast_2d(self.first_lifetime_model.cdf(timeline))  # (m, nb_steps)
            f1m = np.atleast_2d(self.first_lifetime_model.cdf(tm))  # (m, nb_steps - 1)
            y1 = np.atleast_2d(self.func1(timeline))  # (m, nb_steps - 1)
            z1 = np.empty(y1.shape)
            u1 = d * np.insert(f1[:, 1:] - f1m, 0, 1, axis=-1)
            v1 = d[:, :-1] * np.insert(np.diff(f1m), 0, 1, axis=-1)
            z1[:, 0] = y1[:, 0]
            z1[:, 1] = y1[:, 1] + z[:, 0] * u1[:, 1] + z[:, 1] * d[:, 0] * f1m[:, 0]
            for n in range(2, f1.shape[-1]):
                z1[:, n] = (
                    y1[:, n]
                    + z[:, 0] * u1[:, n]
                    + z[:, n] * d[:, 0] * f1m[:, 0]
                    + np.sum(z[:, 1:n][:, ::-1] * v1[:, 1:n], axis=-1)
                )
            return np.squeeze(timeline), np.squeeze(z1)
        return np.squeeze(timeline), np.squeeze(z)


R = TypeVar("R")
P = ParamSpec("P")


def reshape_a0_ar(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        sig = inspect.signature(func)
        ba = sig.bind(*args, **kwargs)
        ba.apply_defaults()
        a0 = ba.arguments.get("a0")
        ar = ba.arguments.get("ar")
        if a0 is not None and isinstance(a0, np.ndarray) and a0.ndim != 2:
            assert is_array_1d(a0)  # typeguard
            ba.arguments["a0"] = to_column_2d_if_1d(a0)
        if ar is not None and isinstance(ar, np.ndarray) and ar.ndim != 2:
            assert is_array_1d(ar)  # typeguard
            ba.arguments["ar"] = to_column_2d_if_1d(ar)
        return func(*ba.args, **ba.kwargs)

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
    """  # noqa: E501

    lifetime_model: ParametricLifetimeModel[()]
    first_lifetime_model: ParametricLifetimeModel[()]
    _different_first_lifetime_model: bool

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        first_lifetime_model: ParametricLifetimeModel[()] | None = None,
    ) -> None:
        super().__init__()
        self.lifetime_model = lifetime_model
        if first_lifetime_model is None:
            self._different_first_lifetime_model = False
            self.first_lifetime_model = self.lifetime_model
        else:
            self._different_first_lifetime_model = True
            self.first_lifetime_model = first_lifetime_model

    @reshape_a0_ar
    def renewal_function(
        self,
        tf: float,
        nb_steps: int,
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        r"""
        The renewal function :math:`m(t) = m_e(t) + m_p(t)`.

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
        a0 : float or np.ndarray, optional
            Initial ages of the assets.
        ar : float or np.ndarray, optional
            Preventive ages of replacements.

        Returns
        -------
        out : tuple of two ndarrays
            A timeline and the corresponding values.

        References
        ----------
        .. [1] Rausand, M., Barros, A., & Hoyland, A. (2020). System Reliability
            Theory: Models, Statistical Methods, and Applications. John Wiley &
            Sons.
        """

        renewal_equation_solver = RenewalEquationSolver(
            get_conditional_lifetime_model(self.lifetime_model, ar=ar),
            get_conditional_lifetime_model(self.first_lifetime_model, ar=ar, a0=a0).cdf,
        )
        return renewal_equation_solver.solve(tf, nb_steps)

    @reshape_a0_ar
    def renewal_density(
        self,
        tf: float,
        nb_steps: int,
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        r"""The renewal density :math:`\omega(t) = m'(t)`.

        .. math::

            \omega(t) = f_1(t) + \int_0^t \omega(t-x) \mathrm{d}F(x)


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
        a0 : float or np.ndarray, optional
            Initial ages of the assets.
        ar : float or np.ndarray, optional
            Preventive ages of replacements.

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
        renewal_equation_solver = RenewalEquationSolver(
            get_conditional_lifetime_model(self.lifetime_model, ar=ar),
            get_conditional_lifetime_model(self.first_lifetime_model, ar=ar, a0=a0).pdf,
        )
        return renewal_equation_solver.solve(tf, nb_steps)

    @reshape_a0_ar
    def expected_number_of_events(
        self,
        tf: float,
        nb_steps: int,
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        r"""
        The expected number of events :math:`m_e(t)`.

        .. math::

            m_e(t) = F(min(t,~a_r)) + \int_0^{+\infty}m_e(t-x)d\hat{F}(x)

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretized the time.
        a0 : float or np.ndarray, optional
            Initial ages of the assets.
        ar : float or np.ndarray, optional
            Preventive ages of replacements.

        Returns
        -------
        out : tuple of two ndarrays
            A timeline and the corresponding values.

        Notes
        -----
        Preventive replacements are not considered as events. Only renewals are. Thus,
        there are not counted.
        """  # noqa: E501

        def F(t: ST | NumpyST | ArrayND[NumpyST]) -> np.float64 | ArrayND[np.float64]:
            _ar = ar if ar is not None else np.inf
            return self.lifetime_model.cdf(np.minimum(t, _ar))

        def F1(
            t: ST | NumpyST | ArrayND[NumpyST],
        ) -> np.float64 | ArrayND[np.float64]:
            left_truncated_model = get_conditional_lifetime_model(
                self.first_lifetime_model, a0=a0
            )
            _ar = ar if ar is not None else np.inf
            _a0 = a0 if a0 is not None else 0.0
            return left_truncated_model.cdf(np.minimum(t, _ar - _a0))

        if self._different_first_lifetime_model or a0 is not None:
            renewal_equation_solver = RenewalEquationSolver(
                get_conditional_lifetime_model(self.lifetime_model, ar=ar),
                F,
                get_conditional_lifetime_model(self.first_lifetime_model, a0=a0, ar=ar),
                F1,
            )
        else:
            renewal_equation_solver = RenewalEquationSolver(
                get_conditional_lifetime_model(self.lifetime_model, ar=ar),
                F,
            )

        return renewal_equation_solver.solve(tf, nb_steps)

    @reshape_a0_ar
    def expected_number_of_preventive_renewals(
        self,
        tf: float,
        nb_steps: int,
        ar: ST | NumpyST | Array1D[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
    ) -> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]]:
        r"""
        The expected number of preventive renewals :math:`m_p(t)`.

        .. math::

            m_p(t) = \mathbb{1}_{t > a_r} \cdot (1 - F(a_r)) + \int_0^{+\infty}m_p(t-x)d\hat{F}(x)

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretized the time.
        ar : float or np.ndarray
            Preventive ages of replacements.
        a0 : float or np.ndarray, optional
            Initial ages of the assets.

        Returns
        -------
        out : tuple of two ndarrays
            A timeline and the corresponding values.

        """  # noqa: E501

        def F(t: ST | NumpyST | ArrayND[NumpyST]) -> np.float64 | ArrayND[np.float64]:
            return (1 - self.lifetime_model.cdf(ar)) * (t > ar)

        def F1(t: ST | NumpyST | ArrayND[NumpyST]) -> np.float64 | ArrayND[np.float64]:
            _a0 = a0 if a0 is not None else 0.0
            first_ar = ar - _a0
            return (
                1
                - get_conditional_lifetime_model(self.first_lifetime_model, a0=a0).cdf(
                    first_ar
                )
            ) * (t > first_ar)

        if self._different_first_lifetime_model or a0 is not None:
            renewal_equation_solver = RenewalEquationSolver(
                get_conditional_lifetime_model(self.lifetime_model, ar=ar),
                F,
                get_conditional_lifetime_model(self.first_lifetime_model, a0=a0, ar=ar),
                F1,
            )
        else:
            renewal_equation_solver = RenewalEquationSolver(
                get_conditional_lifetime_model(self.lifetime_model, ar=ar),
                F,
            )

        return renewal_equation_solver.solve(tf, nb_steps)

    @reshape_a0_ar
    def sample(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
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
        a0 : float or np.ndarray, optional
            Initial ages of the assets.
        ar : float or np.ndarray, optional
            Preventive ages of replacements.
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
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
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
        a0 : float or np.ndarray, optional
            Initial ages of the assets.
        ar : float or np.ndarray, optional
            Preventive ages of replacements.
        seed : int, optional
            Random seed, by default None.

        Returns
        -------
        A dict of time, event, entry and args (covariates)

        """  # noqa: E501

        from ._sample import RenewalProcessIterable

        if self.first_lifetime_model:
            raise ValueError(
                "Calling sample_lifetime_data with first_lifetime_model is ambiguous."  # noqa: E501
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
        self.first_reward = first_reward if first_reward is not None else self.reward
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
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
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
        a0 : float or np.ndarray, optional
            Initial ages of the assets.
        ar : float or np.ndarray, optional
            Preventive ages of replacements.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline and the computed values.

        """

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

        if self._different_first_lifetime_model or a0 is not None:
            renewal_equation_solver = RenewalEquationSolver(
                get_conditional_lifetime_model(self.lifetime_model, ar=ar),
                F,
                get_conditional_lifetime_model(self.first_lifetime_model, a0=a0, ar=ar),
                F1,
            )
        else:
            renewal_equation_solver = RenewalEquationSolver(
                get_conditional_lifetime_model(self.lifetime_model, ar=ar),
                F,
            )

        return renewal_equation_solver.solve(
            tf, nb_steps, discounting_rate=self.discounting_rate
        )

    @reshape_a0_ar
    def asymptotic_expected_total_reward(
        self,
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
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
        a0 : float or np.ndarray, optional
            Initial ages of the assets.
        ar : float or np.ndarray, optional
            Preventive ages of replacements.

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

        if self.first_lifetime_model:
            # Apply delay for the first renewal with a0
            # If no a0 are given, will result in the same solution
            lf1 = np.squeeze(
                get_conditional_lifetime_model(
                    self.first_lifetime_model, a0=a0, ar=ar
                ).ls_integrate(
                    lambda x: self.discounting.factor(x), 0.0, np.inf, deg=100
                )
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
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
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
        a0 : float or np.ndarray, optional
            Initial ages of the assets.
        ar : float or np.ndarray, optional
            Preventive ages of replacements.

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
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
    ) -> np.float64 | Array1D[np.float64]:
        """Asymptotic expected equivalent annual worth.

        Parameters
        ----------
        a0 : float or np.ndarray, optional
            Initial ages of the assets.
        ar : float or np.ndarray, optional
            Preventive ages of replacements.

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
