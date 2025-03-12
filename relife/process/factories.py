from typing import Optional

from relife.core import ParametricModel
from relife.process import (
    RenewalProcess,
    RenewalRewardProcess,
    NonHomogeneousPoissonProcess,
)
from relife.process.nhpp import NonHomogeneousPoissonProcessWithRewards
from relife.rewards import RewardsFunc
from relife.types import Args


def renewal_process(
    model: ParametricModel,
    model_args: tuple[Args, ...] = (),
    nb_assets: int = 1,
    model1: Optional[ParametricModel] = None,
    model1_args: tuple[Args, ...] = (),
    rewards: Optional[RewardsFunc] = None,
    rewards1: Optional[RewardsFunc] = None,
    discounting_rate: Optional[float] = None,
):
    if rewards is None:
        return RenewalProcess(
            model,
            model_args=model_args,
            nb_assets=nb_assets,
            model1=model1,
            model1_args=model1_args,
        )
    else:
        return RenewalRewardProcess(
            model,
            model_args=model_args,
            nb_assets=nb_assets,
            model1=model1,
            model1_args=model1_args,
            rewards=rewards,
            rewards1=rewards1,
            discounting_rate=discounting_rate,
        )


def poisson_process(
    model: ParametricModel,
    model_args: tuple[Args, ...] = (),
    nb_assets: int = 1,
    rewards: Optional[RewardsFunc] = None,
    discounting_rate: Optional[float] = None,
):

    if rewards is None:
        return NonHomogeneousPoissonProcess(
            model,
            model_args=model_args,
            nb_assets=nb_assets,
        )
    else:
        return NonHomogeneousPoissonProcessWithRewards(
            model,
            rewards,
            model_args=model_args,
            nb_assets=nb_assets,
            discounting_rate=discounting_rate,
        )
