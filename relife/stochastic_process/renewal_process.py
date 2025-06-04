from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Generic, Optional, TypeVar, Union, TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife import ParametricModel
from relife.economic import (
    Discounting,
    ExponentialDiscounting,
    Reward,
)
from relife.lifetime_model import (
    FrozenAgeReplacementModel,
    FrozenLeftTruncatedModel,
    FrozenLifetimeRegression,
    LifetimeDistribution,
)

if TYPE_CHECKING:
    from relife.economic import Reward

M = TypeVar(
    "M",
    bound=Union[LifetimeDistribution, FrozenLifetimeRegression, FrozenAgeReplacementModel, FrozenLeftTruncatedModel],
)
R = TypeVar("R", bound=Reward)


class LifetimeFitArg(TypedDict):
    time : NDArray[np.float64]
    event : NDArray[np.bool_]
    entry : NDArray[np.float64]
    args : tuple[NDArray[np.float64], ...]


class RenewalProcess(ParametricModel, Generic[M]):
    """Renewal process.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.

    first_lifetime_model : any lifetime distribution or frozen lifetime model, optional
        A lifetime model for the first renewal (delayed renewal process). It is None by default
    """

    def __init__(
        self,
        lifetime_model: M,
        first_lifetime_model: Optional[M] = None,
    ):
        super().__init__()

        self.lifetime_model = lifetime_model
        self.first_lifetime_model = first_lifetime_model
        self.count_data = None


    def _make_timeline(self, tf: float, nb_steps: int) -> NDArray[np.float64]:
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
        args_nb_assets = getattr(self.lifetime_model, "args_nb_assets", 1)  # default 1 for LifetimeDistribution case
        if args_nb_assets > 1:
            timeline = np.tile(timeline, (args_nb_assets, 1))
        return timeline  # (nb_steps,) or (m, nb_steps)

    def renewal_function(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""The renewal function.

        Parameters
        ----------
        tf : float
            Time horizon. The renewal function will be computed up until this calendar time.
        nb_steps : int
            The number of steps used to compute the renewal function.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the renewal function and its corresponding values at each
            step of the timeline.

        Notes
        -----
        The expected total number of renewals is computed  by solving the
        renewal equation:

        .. math::

            m(t) = F_1(t) + \int_0^t m(t-x) \mathrm{d}F(x)

        where:

        - :math:`m` is the renewal function,
        - :math:`F` is the cumulative distribution function of the underlying
          lifetime model,
        - :math:`F_1` is the cumulative distribution function of the underlying
          lifetime model for the fist renewal in the case of a delayed renewal
          process.

        References
        ----------
        .. [1] Rausand, M., Barros, A., & Hoyland, A. (2020). System Reliability
            Theory: Models, Statistical Methods, and Applications. John Wiley &
            Sons.
        """

        timeline = self._make_timeline(tf, nb_steps)  # (nb_steps,) or (m, nb_steps)
        return timeline, renewal_equation_solver(
            timeline,
            self.lifetime_model,
            self.lifetime_model.cdf if self.first_lifetime_model is None else self.first_lifetime_model.cdf,
        )

    def renewal_density(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""The renewal density.

        Parameters
        ----------
        tf : float
            Time horizon. The renewal density will be computed up until this calendar time.
        nb_steps : int
            The number of steps used to compute the renewal density.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the renewal density and its corresponding values at each
            step of the timeline.

        Notes
        -----
        The renewal density is the derivative of the renewal function with
        respect to time. It is computed by solving the renewal equation:

        .. math::

            \mu(t) = f_1(t) + \int_0^t \mu(t-x) \mathrm{d}F(x)

        where:

        - :math:`\mu` is the renewal function,
        - :math:`F` is the cumulative distribution function of the underlying
          lifetime model,
        - :math:`f_1` is the probability density function of the underlying
          lifetime model for the fist renewal in the case of a delayed renewal
          process.

        References
        ----------
        .. [1] Rausand, M., Barros, A., & Hoyland, A. (2020). System Reliability
            Theory: Models, Statistical Methods, and Applications. John Wiley &
            Sons.
        """
        timeline = self._make_timeline(tf, nb_steps)  #  (nb_steps,) or (m, nb_steps)
        return timeline, renewal_equation_solver(
            timeline,
            self.lifetime_model,
            self.lifetime_model.pdf if self.first_lifetime_model is None else self.first_lifetime_model.pdf,
        )

    def sample(
        self,
        tf: float,
        t0: float = 0.0,
        size: int | tuple[int] | tuple[int, int] = 1,
        seed: Optional[int] = None,
    ) -> RenewalProcessSample:
        """Renewal data sampling.

        This function will generate sampling data insternally. These data

        Parameters
        ----------
        tf : float
            Time at the end of the observation.
        t0 : float, default 0
            Time at the beginning of the observation.
        size : int or tuple of 2 int
            Size of the sample
        seed : int, optional
            Random seed, by default None.

        """

        from .sample import RenewalProcessIterable

        iterable = RenewalProcessIterable(self, size, (t0, tf), seed=seed)
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("nb_renewal", "asset_id", "sample_id"))
        return RenewalProcessSample(t0, tf, struct_array)

    def generate_lifetime_data(
        self,
        tf: float,
        t0: float = 0.0,
        size: int | tuple[int] | tuple[int, int] = 1,
        seed: Optional[int] = None,
    ) -> LifetimeFitArg:
        from .sample import RenewalProcessIterable

        if self.first_lifetime_model is not None and self.first_lifetime_model != self.lifetime_model:
            from relife.lifetime_model import FrozenLeftTruncatedModel

            if (
                isinstance(self.first_lifetime_model, FrozenLeftTruncatedModel)
                and self.first_lifetime_model.unfreeze() == self.lifetime_model
            ):
                pass
            else:
                raise ValueError(
                    f"Calling sample_lifetime_data with lifetime_model different from first_lifetime_model is ambiguous."
                )
        iterable = RenewalProcessIterable(self, size, (t0, tf), seed=seed)
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("nb_renewal", "asset_id", "sample_id"))

        nb_assets = int(np.max(struct_array["asset_id"])) + 1
        args_2d = tuple((np.atleast_2d(arg) for arg in getattr(self.lifetime_model, "frozen_args", ())))
        broadcasted_args = tuple((np.broadcast_to(arg, (nb_assets, arg.shape[-1])) for arg in args_2d))
        tuple_args_arr = tuple((np.take(np.asarray(arg), struct_array["asset_id"]) for arg in broadcasted_args))

        returned_dict = {
            "time" : struct_array["time"].copy(),
            "event" : struct_array["event"].copy(),
            "entry" : struct_array["entry"].copy(),
            "args" : tuple_args_arr
        }
        return returned_dict



class RenewalRewardProcess(RenewalProcess[M], Generic[M, R]):

    def __init__(
        self,
        lifetime_model: M,
        reward: R,
        discounting_rate: float = 0.0,
        first_lifetime_model: Optional[M] = None,
        first_reward: Optional[R] = None,
    ):
        super().__init__(lifetime_model, first_lifetime_model)
        self.reward = reward
        self.first_reward = first_reward if first_reward is not None else reward
        self.discounting = ExponentialDiscounting(discounting_rate)

    @property
    def discounting_rate(self) -> float:
        return self.discounting.rate

    @override
    def _make_timeline(self, tf: float, nb_steps: int) -> NDArray[np.float64]:
        # control with reward too
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
        args_nb_assets = getattr(self.lifetime_model, "args_nb_assets", 1)  # default 1 for LifetimeDistribution case
        if args_nb_assets > 1:
            timeline = np.tile(timeline, (args_nb_assets, 1))
        elif self.reward.ndim == 2:  # elif because we consider that if m > 1 in frozen_model, in reward it is 1 or m
            timeline = np.tile(timeline, (self.reward.size, 1))
        return timeline  # (nb_steps,) or (m, nb_steps)

    def expected_total_reward(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = self._make_timeline(tf, nb_steps)  #  (nb_steps,) or (m, nb_steps)
        z = renewal_equation_solver(
            timeline,
            self.lifetime_model,
            lambda t: self.lifetime_model.ls_integrate(
                lambda x: self.reward.sample(x) * self.discounting.factor(x), np.zeros_like(t), t, deg=15
            ),  # reward partial expectation
            discounting=self.discounting,
        )
        if self.first_lifetime_model is not None:
            return delayed_renewal_equation_solver(
                timeline,
                z,
                self.first_lifetime_model,
                lambda t: self.first_lifetime_model.ls_integrate(
                    lambda x: self.first_reward.sample(x) * self.discounting.factor(x),
                    np.zeros_like(t),
                    timeline,
                    deg=15,
                ),  # reward partial expectation
                discounting=self.discounting,
            )
        return timeline, z  # (nb_steps,) or (m, nb_steps)

    def asymptotic_expected_total_reward(
        self,
    ) -> NDArray[np.float64]:
        lf = self.lifetime_model.ls_integrate(lambda x: self.discounting.factor(x), 0.0, np.inf, deg=15)
        ly = self.lifetime_model.ls_integrate(
            lambda x: self.discounting.factor(x) * self.reward.sample(x), 0.0, np.inf, deg=15
        )
        z = ly / (1 - lf)  # () or (m, 1)
        if self.first_lifetime_model is not None:
            lf1 = self.first_lifetime_model.ls_integrate(lambda x: self.discounting.factor(x), 0.0, np.inf, deg=15)
            ly1 = self.first_lifetime_model.ls_integrate(
                lambda x: self.discounting.factor(x) * self.first_reward.sample(x), 0.0, np.inf, deg=15
            )
            z = ly1 + z * lf1  # () or (m, 1)
        return z  # () or (m, 1)

    def expected_equivalent_annual_worth(
        self, tf: float, nb_steps: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline, z = self.expected_total_reward(tf, nb_steps)  # (nb_steps,) or (m, nb_steps)
        af = self.discounting.annuity_factor(timeline)  # (nb_steps,) or (m, nb_steps)
        q = z / (af + 1e-6)  # # (nb_steps,) or (m, nb_steps) avoid zero division
        if self.first_lifetime_model is not None:
            q0 = self.first_reward.sample(0.0) * self.first_lifetime_model.pdf(0.0)
        else:
            q0 = self.reward.sample(0.0) * self.lifetime_model.pdf(0.0)
        # q0 : () or (m, 1)
        q0 = np.broadcast_to(q0, af.shape)  # (), (nb_steps,) or (m, nb_steps)
        return timeline, np.where(af == 0, q0, q)

    def asymptotic_expected_equivalent_annual_worth(self) -> NDArray[np.float64]:
        if self.discounting_rate == 0.0:
            return (
                self.lifetime_model.ls_integrate(lambda x: self.reward.sample(x), 0.0, np.inf, deg=15)
                / self.lifetime_model.mean()
            )  # () or (m, 1)
        return self.discounting_rate * self.asymptotic_expected_total_reward()  # () or (m, 1)

    @override
    def sample(
        self,
        tf: float,
        t0: float = 0.0,
        size: int | tuple[int] | tuple[int, int] = 1,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> RenewalRewardProcessSample:
        from .sample import RenewalProcessIterable

        iterable = RenewalProcessIterable(self, size, (t0, tf), seed=seed)
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("nb_renewal", "asset_id", "sample_id"))
        return RenewalRewardProcessSample(t0, tf, struct_array)


def renewal_equation_solver(
    timeline: NDArray[np.float64],
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel,
    evaluated_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    discounting: Optional[Discounting] = None,
) -> NDArray[np.float64]:

    # timeline : (nb_steps,) or (m, nb_steps)
    tm = 0.5 * (timeline[..., 1:] + timeline[..., :-1])  # (nb_steps - 1,) or (m, nb_steps - 1)
    f = lifetime_model.cdf(timeline)  # (nb_steps,) or (m, nb_steps)
    fm = lifetime_model.cdf(tm)  # (nb_steps - 1,) or (m, nb_steps - 1)
    y = evaluated_func(timeline)  # (nb_steps,) or (m, nb_steps)

    if y.shape != f.shape:
        raise ValueError("Invalid shape between model and evaluated_func")

    if discounting is not None:
        d = discounting.factor(timeline)  # (nb_steps,) or (m, nb_steps)
    else:
        d = np.ones_like(f)
    z = np.empty(y.shape)
    u = d * np.insert(f[..., 1:] - fm, 0, 1, axis=-1)
    v = d[..., :-1] * np.insert(np.diff(fm), 0, 1, axis=-1)
    q0 = 1 / (1 - d[..., 0] * fm[..., 0])
    z[..., 0] = y[..., 0]
    z[..., 1] = q0 * (y[..., 1] + z[..., 0] * u[..., 1])
    for n in range(2, f.shape[-1]):
        z[..., n] = q0 * (y[..., n] + z[..., 0] * u[..., n] + np.sum(z[..., 1:n][..., ::-1] * v[..., 1:n], axis=-1))
    return z


def delayed_renewal_equation_solver(
    timeline: NDArray[np.float64],
    z: NDArray[np.float64],
    first_lifetime_model: (
        LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel | FrozenLeftTruncatedModel
    ),
    evaluated_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    discounting: Optional[ExponentialDiscounting] = None,
) -> NDArray[np.float64]:
    # timeline : (nb_steps,) or (m, nb_steps)
    tm = 0.5 * (timeline[..., 1:] + timeline[..., :-1])  # (nb_steps - 1,) or (m, nb_steps - 1)
    f1 = first_lifetime_model.cdf(timeline)  # (nb_steps,) or (m, nb_steps)
    f1m = first_lifetime_model.cdf(tm)  # (nb_steps - 1,) or (m, nb_steps - 1)
    y1 = evaluated_func(timeline)  # (nb_steps,) or (m, nb_steps - 1)
    if discounting is not None:
        d = discounting.factor(timeline)  # (nb_steps,) or (m, nb_steps - 1)
    else:
        d = np.ones_like(f1)
    z1 = np.empty(y1.shape)
    u1 = d * np.insert(f1[..., 1:] - f1m, 0, 1, axis=-1)
    v1 = d[..., :-1] * np.insert(np.diff(f1m), 0, 1, axis=-1)
    z1[..., 0] = y1[..., 0]
    z1[..., 1] = y1[..., 1] + z[..., 0] * u1[..., 1] + z[..., 1] * d[..., 0] * f1m[..., 0]
    for n in range(2, f1.shape[-1]):
        z1[..., n] = (
            y1[..., n]
            + z[..., 0] * u1[..., n]
            + z[..., n] * d[..., 0] * f1m[..., 0]
            + np.sum(z[..., 1:n][..., ::-1] * v1[..., 1:n], axis=-1)
        )
    return z1


@dataclass
class CountDataSample:
    t0: float
    tf: float
    struct_array: NDArray[np.void]

    def select(self, sample_id: Optional[int] = None, asset_id: Optional[int] = None) -> CountDataSample:
        mask: NDArray[np.bool_] = np.ones_like(self.struct_array, dtype=np.bool_)
        if sample_id is not None:
            mask = mask & np.isin(self.struct_array["sample_id"], sample_id)
        if asset_id is not None:
            mask = mask & np.isin(self.struct_array["asset_id"], asset_id)
        struct_subarray = self.struct_array[mask].copy()
        return replace(self, t0=self.t0, tf=self.tf, struct_array=struct_subarray)


@dataclass
class RenewalProcessSample(CountDataSample):

    @staticmethod
    def _nb_events(selection: RenewalProcessSample) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sort = np.argsort(selection.struct_array["timeline"])
        timeline = selection.struct_array["timeline"][sort]
        counts = np.ones_like(timeline)

        timeline = np.insert(timeline, 0, selection.t0)
        counts = np.insert(counts, 0, 0)
        counts[timeline == selection.tf] = 0
        return timeline, np.cumsum(counts)

    def nb_events(self, sample_id: int, asset_id: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        selection = self.select(sample_id=sample_id, asset_id=asset_id)
        return RenewalProcessSample._nb_events(selection)

    def mean_nb_events(self, asset_id: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        selection = self.select(asset_id=asset_id)
        timeline, counts = RenewalProcessSample._nb_events(selection)
        nb_sample = len(np.unique(selection.struct_array["sample_id"]))
        return timeline, counts / nb_sample


@dataclass
class RenewalRewardProcessSample(RenewalProcessSample):

    @staticmethod
    def _total_rewards(selection: RenewalRewardProcessSample) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sort = np.argsort(selection.struct_array["timeline"])
        timeline = selection.struct_array["timeline"][sort]
        reward = selection.struct_array["reward"][sort]

        timeline = np.insert(timeline, 0, selection.t0)
        reward = np.insert(reward, 0, 0)
        reward[timeline == selection.tf] = 0

        return timeline, np.cumsum(reward)

    def total_rewards(self, sample_id: int, asset_id: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        selection = self.select(sample_id=sample_id, asset_id=asset_id)
        return RenewalRewardProcessSample._total_rewards(selection)

    def mean_total_rewards(self, asset_id: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        selection = self.select(asset_id=asset_id)
        timeline, rewards = RenewalRewardProcessSample._nb_events(selection)
        nb_sample = len(np.unique(selection.struct_array["sample_id"]))
        return timeline, rewards / nb_sample
