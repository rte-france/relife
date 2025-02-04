from typing import Optional

import numpy as np
from numpy.typing import NDArray

from relife.data import RenewalRewardData
from relife.discountings import exponential_discounting
from relife.generator import lifetimes_rewards_generator
from relife.model import LeftTruncatedModel, LifetimeModel
from relife.renewal import RenewalRewardProcess, reward_partial_expectation
from relife.rewards import run_to_failure_cost
from relife.typing import Model1Args, ModelArgs, Policy


class OneCycleRunToFailure(Policy):
    r"""One cyle run-to-failure policy

    A policy for running assets to failure within one cycle.

    Parameters
    ----------
    model : LifetimeModel
        The lifetime model of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    discounting_rate : float, default is 0.
        The discounting rate.
    model_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the underlying
        lifetime model of the process.
    nb_assets : int, optional
        Number of assets (default is 1).
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.

    """

    reward = run_to_failure_cost
    discounting = exponential_discounting
    model1 = None

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        cf: float | NDArray[np.float64],
        *,
        discounting_rate: float = 0.0,
        model_args: ModelArgs = (),
        nb_assets: int = 1,
        a0: Optional[float | NDArray[np.float64]] = None,
    ) -> None:
        if a0 is not None:
            model = LeftTruncatedModel(model)
            model_args = (a0, *model_args)
        self.model = model
        self.nb_assets = nb_assets
        self.cf = cf
        self.discounting_rate = discounting_rate
        self.model_args = model_args

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        """The expected total cost.

        Parameters
        ----------
        timeline : ndarray
            Timeline of points where the function is evaluated

        Returns
        -------
        ndarray
            The expected total cost for each asset along the timeline
        """
        return reward_partial_expectation(
            timeline,
            self.model,
            run_to_failure_cost,
            exponential_discounting,
            model_args=self.model_args,
            reward_args=(self.cf,),
            discounting_args=(self.discounting_rate,),
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        """
        The asymptotic expected total cost.

        Returns
        -------
        ndarray
            The asymptotic expected total cost for each asset.
        """
        return self.expected_total_cost(np.array(np.inf))

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64], dt: float = 1.0
    ) -> NDArray[np.float64]:
        """The expected equivalent annual cost.

        Parameters
        ----------
        timeline : ndarray
            Timeline of points where the function is evaluated
        dt : float, default 1.0
            The length of the first period before discounting

        Returns
        -------
        ndarray
            The expected equivalent annual cost until each time point
        """

        f = (
            lambda x: run_to_failure_cost(x, self.cf)
            * exponential_discounting.factor(x, self.discounting_rate)
            / exponential_discounting.annuity_factor(x, self.discounting_rate)
        )
        mask = timeline < dt
        q0 = self.model.cdf(dt, *self.model_args) * f(dt)
        return q0 + np.where(
            mask, 0, self.model.ls_integrate(f, dt, timeline, *self.model_args)
        )

    def asymptotic_expected_equivalent_annual_cost(
        self,
    ) -> NDArray[np.float64]:
        """
        The asymptotic expected equivalent annual cost.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost.
        """
        return self.expected_equivalent_annual_cost(np.array(np.inf))

    def sample(
        self,
        nb_samples: int,
        seed: Optional[int] = None,
    ) -> RenewalRewardData:
        """Sample simulation .

        Parameters
        ----------
        nb_samples : int
            Number of samples generated
        seed : int, optional
            Sample seed. Usefull to fix random generation and reproduce results

        Returns
        -------
        RenewalRewardData
            Iterable object that encapsulates results with additional functions
        """
        generator = lifetimes_rewards_generator(
            self.model,
            self.reward,
            self.discounting,
            nb_samples,
            self.nb_assets,
            np.inf,
            model_args=self.model_args,
            reward_args=(self.cf,),
            discounting_args=(self.discounting_rate,),
            seed=seed,
        )
        _lifetimes, _event_times, _total_rewards, _events, still_valid = next(generator)
        assets_index, samples_index = np.where(still_valid)
        assets_index.astype(np.int64)
        samples_index.astype(np.int64)
        lifetimes = _lifetimes[still_valid]
        event_times = _event_times[still_valid]
        total_rewards = _total_rewards[still_valid]
        events = _events[still_valid]
        order = np.zeros_like(lifetimes)

        return RenewalRewardData(
            samples_index,
            assets_index,
            order,
            event_times,
            lifetimes,
            events,
            self.model_args,
            False,
            total_rewards,
        )


class RunToFailure(Policy):
    r"""Run-to-failure renewal policy.

    Renewal reward process where assets are replaced on failure with costs
    :math:`c_f`.

    Parameters
    ----------
    model : LifetimeModel
        The lifetime model of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    discounting_rate : float, default is 0.
        The discounting rate.
    model_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the underlying
        lifetime model of the process.
    nb_assets : int, optional
        Number of assets (default is 1).
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.
    model1 : LifetimeModel, optional
        The lifetime model used for the cycle of replacements. When one adds
        `model1`, we assume that `model1` is different from `model` meaning
        the underlying survival probabilities behave differently for the first
        cycle
    model1_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the lifetime
        model of the first cycle of replacements.

    References
    ----------
    .. [1] Van der Weide, J. A. M., & Van Noortwijk, J. M. (2008). Renewal
        theory with exponential and hyperbolic discounting. Probability in
        the Engineering and Informational Sciences, 22(1), 53-74.
    """

    reward = run_to_failure_cost
    discounting = exponential_discounting

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        cf: float | NDArray[np.float64],
        *,
        discounting_rate: float = 0.0,
        model_args: ModelArgs = (),
        nb_assets: int = 1,
        a0: Optional[float | NDArray[np.float64]] = None,
        model1: Optional[LifetimeModel[*Model1Args]] = None,
        model1_args: Model1Args = (),
    ) -> None:

        if a0 is not None:
            if model1 is not None:
                raise ValueError("model1 and a0 can't be set together")
            model1 = LeftTruncatedModel(model)
            model1_args = (a0, *model_args)

        self.model = model
        self.model1 = model1

        self.model_args = model_args
        self.cf = cf
        self.discounting_rate = discounting_rate

        self.model_args = model_args
        self.model1_args = model1_args

        self.nb_assets = nb_assets

        # if Policy is parametrized, set the underlying renewal reward process
        # note the rewards are the same for the first cycle and the rest of the process
        self.rrp = RenewalRewardProcess(
            self.model,
            self.reward,
            nb_assets=self.nb_assets,
            model_args=self.model_args,
            reward_args=(self.cf,),
            discounting_rate=self.discounting_rate,
            model1=self.model1,
            model1_args=self.model1_args,
            reward1=self.reward,
            reward1_args=(self.cf,),
        )

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        """The expected total cost.

        Parameters
        ----------
        timeline : ndarray
            Timeline of points where the function is evaluated

        Returns
        -------
            The expected total cost for each asset along the timeline
        """
        return self.rrp.expected_total_reward(timeline)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        """
        The asymptotic expected total cost.

        Returns
        -------
        ndarray
            The asymptotic expected total cost for each asset.
        """
        return self.rrp.asymptotic_expected_total_reward()

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The expected equivalent annual cost.

        Parameters
        ----------
        timeline : ndarray
            Timeline of points where the function is evaluated

        Returns
        -------
        ndarray
            The expected equivalent annual cost until each time point
        """
        return self.rrp.expected_equivalent_annual_cost(timeline)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        """
        The asymptotic expected equivalent annual cost.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost.
        """
        return self.rrp.asymptotic_expected_equivalent_annual_cost()

    def sample(
        self,
        nb_samples: int,
        end_time: float,
        seed: Optional[int] = None,
    ) -> RenewalRewardData:
        """Sample simulation .

        Parameters
        ----------
        nb_samples : int
            Number of samples generated
        end_time : float
            End of the observation period. It is the upper bound of the cumulative generated lifetimes.
        seed : int, optional
            Sample seed. Usefull to fix random generation and reproduce results

        Returns
        -------
        RenewalRewardData
            Iterable object that encapsulates results with additional functions
        """
        return self.rrp.sample(nb_samples, end_time, seed=seed)

    def expected_number_of_replacements(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def expected_number_of_failures(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def expected_number_of_preventive_replacements(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
