from typing import TypedDict, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

VariadicArgs = TypeVarTuple("VariadicArgs")

# tuple consisting of zero or more NDArray[np.float64]
ModelArgs = tuple[NDArray[np.float64], ...]
Model1Args = tuple[NDArray[np.float64], ...]
RewardArgs = tuple[NDArray[np.float64], ...]
Reward1Args = tuple[NDArray[np.float64], ...]
DiscountArgs = tuple[NDArray[np.float64], ...]


class RenewalProcessArgs(TypedDict):
    model: ModelArgs
    model1: Model1Args


class RenewalRewardProcessArgs(TypedDict):
    model: ModelArgs
    model1: Model1Args
    reward: RewardArgs
    reward1: Reward1Args
    discount: DiscountArgs


class PolicyArgs(TypedDict, total=False):
    model: ModelArgs
    model1: Model1Args
    reward: RewardArgs
    reward1: Reward1Args
    discount: DiscountArgs
