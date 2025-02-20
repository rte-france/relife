from typing import Optional

import numpy as np
from numpy.typing import NDArray

from relife.data import RenewalRewardData
from relife.core.descriptors import ShapedArgs
from relife.core.discounting import exponential_discounting
from relife.generator import lifetimes_rewards_generator
from relife.core.nested_model import LeftTruncatedModel
from relife.core.model import LifetimeModel
from .docstrings import (
    ETC_DOCSTRING,
    EEAC_DOCSTRING,
    ASYMPTOTIC_ETC_DOCSTRING,
    ASYMPTOTIC_EEAC_DOCSTRING,
)
from relife.process.renewal import RenewalRewardProcess, reward_partial_expectation
from relife.types import TupleArrays, Policy


def run_to_failure_cost(
    lifetimes: NDArray[np.float64], cf: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.ones_like(lifetimes) * cf


class OneCycleRunToFailure(Policy):
    r"""One cyle run-to-failure policy

    A policy for running assets to failure within one cycle.

    Parameters
    ----------
    model : LifetimeModel
        The lifetime core of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    discounting_rate : float, default is 0.
        The discounting rate.
    period_before_discounting: float, default is 1.
        The length of the first period before discounting.
    model_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the underlying
        lifetime core of the process.
    nb_assets : int, optional
        Number of assets (default is 1).
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.
    """

    model1 = None

    cf = ShapedArgs()
    a0 = ShapedArgs()
    model_args = ShapedArgs(astuple=True)

    def __init__(
        self,
        model: LifetimeModel[*TupleArrays],
        cf: float | NDArray[np.float64],
        *,
        discounting_rate: float = 0.0,
        period_before_discounting: float = 1.0,
        model_args: TupleArrays = (),
        nb_assets: int = 1,
        a0: Optional[float | NDArray[np.float64]] = None,
    ) -> None:
        self.nb_assets = nb_assets
        if a0 is not None:
            model = LeftTruncatedModel(model)
            model_args = (a0, *model_args)
        self.model = model
        self.cf = cf
        self.discounting_rate = discounting_rate
        if period_before_discounting == 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting
        self.model_args = model_args

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            self.model,
            run_to_failure_cost,
            model_args=self.model_args,
            reward_args=(self.cf,),
            discounting_rate=self.discounting_rate,
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.expected_total_cost(np.array(np.inf))

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        f = (
            lambda x: run_to_failure_cost(x, self.cf)
            * exponential_discounting.factor(x, self.discounting_rate)
            / exponential_discounting.annuity_factor(x, self.discounting_rate)
        )
        mask = timeline < self.period_before_discounting
        q0 = self.model.cdf(self.period_before_discounting, *self.model_args) * f(
            self.period_before_discounting
        )
        return np.squeeze(
            q0
            + np.where(
                mask,
                0,
                self.model.ls_integrate(
                    f, self.period_before_discounting, timeline, *self.model_args
                ),
            )
        )

    def asymptotic_expected_equivalent_annual_cost(
        self,
    ) -> NDArray[np.float64]:
        return self.expected_equivalent_annual_cost(np.array(np.inf))

    def sample(
        self,
        nb_samples: int,
        seed: Optional[int] = None,
        maxsample: int = 1e5,
    ) -> RenewalRewardData:
        r"""
        Samples lifetimes and rewards.

        Parameters
        ----------
            nb_samples : int
                The number of samples to be generated.
            seed : int or None, optional
                An optional seed for random number generation to ensure reproducibility. Defaults to None.

        Returns
        -------
        RenewalRewardData
            The resulting :py:class:`~relife.data.counting.CountData` object which encapsulates the sampled data.

        .. note::

            The sampling process stops after the first iteration as only one cycle of replacement is considered.
        """
        generator = lifetimes_rewards_generator(
            self.model,
            run_to_failure_cost,
            nb_samples,
            self.nb_assets,
            np.inf,
            model_args=self.model_args,
            reward_args=(self.cf,),
            discounting_rate=self.discounting_rate,
            seed=seed,
        )
        samples_ids, assets_ids, lifetimes, event_times, total_rewards, events = next(
            generator
        )
        if len(samples_ids) > maxsample:
            raise ValueError(
                "Max number of sample has been reach : 1e5. Modify maxsample or set different arguments"
            )

        return RenewalRewardData(
            samples_ids,
            assets_ids,
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
        The lifetime core of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    discounting_rate : float, default is 0.
        The discounting rate.
    model_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the underlying
        lifetime core of the process.
    nb_assets : int, optional
        Number of assets (default is 1).
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.
    model1 : LifetimeModel, optional
        The lifetime core used for the cycle of replacements. When one adds
        `model1`, we assume that `model1` is different from `core` meaning
        the underlying survival probabilities behave differently for the first
        cycle
    model1_args : ModelArgs, optional
        ModelArgs is a tuple of zero or more ndarray required by the lifetime
        core of the first cycle of replacements.

    References
    ----------
    .. [1] Van der Weide, J. A. M., & Van Noortwijk, J. M. (2008). Renewal
        theory with exponential and hyperbolic discounting. Probability in
        the Engineering and Informational Sciences, 22(1), 53-74.
    """

    cf = ShapedArgs()
    a0 = ShapedArgs()
    model_args = ShapedArgs(astuple=True)
    model1_args = ShapedArgs(astuple=True)

    def __init__(
        self,
        model: LifetimeModel[*TupleArrays],
        cf: float | NDArray[np.float64],
        *,
        discounting_rate: float = 0.0,
        model_args: TupleArrays = (),
        nb_assets: int = 1,
        a0: Optional[float | NDArray[np.float64]] = None,
        model1: Optional[LifetimeModel[*TupleArrays]] = None,
        model1_args: TupleArrays = (),
    ) -> None:

        self.nb_assets = nb_assets

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

        # if Policy is parametrized, set the underlying renewal reward process
        # note the rewards are the same for the first cycle and the rest of the process
        self.process = RenewalRewardProcess(
            self.model,
            run_to_failure_cost,
            nb_assets=self.nb_assets,
            model_args=self.model_args,
            reward_args=(self.cf,),
            discounting_rate=self.discounting_rate,
            model1=self.model1,
            model1_args=self.model1_args,
            reward1=run_to_failure_cost,
            reward1_args=(self.cf,),
        )

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.process.expected_total_reward(timeline)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.process.asymptotic_expected_total_reward()

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.process.expected_equivalent_annual_cost(timeline)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        return self.process.asymptotic_expected_equivalent_annual_cost()

    def sample(
        self,
        nb_samples: int,
        end_time: float,
        seed: Optional[int] = None,
    ) -> RenewalRewardData:
        """
        Samples lifetimes and rewards.

        Parameters
        ----------
            nb_samples : int
                The number of samples to be generated.
            end_time : float
                The temporal limit for the sampling operation.
            seed : int or None, optional
                An optional seed for random number generation to ensure reproducibility. Defaults to None.

        Returns
        -------
        RenewalRewardData
            The resulting :py:class:`~relife.data.counting.CountData` object which encapsulates the sampled data.
        """
        return self.process.sample(nb_samples, end_time, seed=seed)


class _RunToFailurePolicy:
    def __init__(self, model: LifetimeModel[*TupleArrays]):
        self.model = model
        self.policy = None
        self.fitted = False

    def load(self, /, *args, nature: str = "default", **kwargs):
        if self.policy is not None:
            raise ValueError("AgeReplacementPolicy is already load")
        elif type == "default":
            self.policy = RunToFailure(self.model, *args, **kwargs)
        elif type == "one_cycle":
            self.policy = OneCycleRunToFailure(self.model, *args, **kwargs)
        else:
            raise ValueError(f"Uncorrect nature {nature}")

    def __getattr__(self, item):
        class_name = type(self).__name__
        if self.policy is None:
            raise ValueError("No policy as been loaded")
        if item in super().__getattribute__("policy"):
            return super().__getattribute__("policy")[item]
        raise AttributeError(f"{class_name} has no attribute/method {item}")


OneCycleRunToFailure.expected_total_cost.__doc__ = ETC_DOCSTRING
OneCycleRunToFailure.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING
OneCycleRunToFailure.asymptotic_expected_total_cost.__doc__ = ASYMPTOTIC_ETC_DOCSTRING
OneCycleRunToFailure.asymptotic_expected_equivalent_annual_cost.__doc__ = (
    ASYMPTOTIC_EEAC_DOCSTRING
)

RunToFailure.expected_total_cost.__doc__ = ETC_DOCSTRING
RunToFailure.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING
RunToFailure.asymptotic_expected_total_cost.__doc__ = ASYMPTOTIC_ETC_DOCSTRING
RunToFailure.asymptotic_expected_equivalent_annual_cost.__doc__ = (
    ASYMPTOTIC_EEAC_DOCSTRING
)
