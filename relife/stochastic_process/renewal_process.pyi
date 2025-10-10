from typing import Optional, TypeAlias, TypedDict, Union

import numpy as np
from numpy.typing import NDArray as NDArray
from typing_extensions import override

from relife._typing import _Any_Number
from relife.base import FrozenParametricModel, ParametricModel
from relife.economic import ExponentialDiscounting, Reward
from relife.lifetime_model import ParametricLifetimeModel

from ._sample import RenewalProcessSample, RenewalRewardProcessSample

class LifetimeFitArg(TypedDict):
    time: NDArray[np.float64]
    event: NDArray[np.bool_]
    entry: NDArray[np.float64]
    args: tuple[NDArray[np.float64], ...]

# any ParametricLifetimeModel with no args (LifetimeDistribution : OK) OR any Frozen-ParametricLifetimeModel with at least one arg
_FrozenLike_ParametricLifetimeModel: TypeAlias = Union[
    ParametricLifetimeModel[()],
    FrozenParametricModel[ParametricLifetimeModel[*tuple[_Any_Number, *tuple[_Any_Number, ...]]]],
]
_Timeline: TypeAlias = NDArray[np.float64]
_Expected_Values: TypeAlias = NDArray[np.float64]
_Asymptotic_Expected_Values: TypeAlias = Union[np.float64, NDArray[np.float64]]

class RenewalProcess(ParametricModel):

    lifetime_model: _FrozenLike_ParametricLifetimeModel
    first_lifetime_model: Optional[_FrozenLike_ParametricLifetimeModel]

    def __init__(
        self,
        lifetime_model: _FrozenLike_ParametricLifetimeModel,
        first_lifetime_model: Optional[_FrozenLike_ParametricLifetimeModel] = None,
    ) -> None: ...
    def _make_timeline(self, tf: float, nb_steps: int) -> _Timeline: ...
    def renewal_function(self, tf: float, nb_steps: int) -> tuple[_Timeline, _Expected_Values]: ...
    def renewal_density(self, tf: float, nb_steps: int) -> tuple[_Timeline, _Expected_Values]: ...
    def sample(self, size: int, tf: float, t0: float = 0.0, seed: Optional[int] = None) -> RenewalProcessSample: ...
    def generate_failure_data(
        self, size: int, tf: float, t0: float = 0.0, seed: Optional[int] = None
    ) -> LifetimeFitArg: ...

class RenewalRewardProcess(RenewalProcess):
    reward: Reward
    first_reward: Optional[Reward]
    discounting: ExponentialDiscounting
    def __init__(
        self,
        lifetime_model: _FrozenLike_ParametricLifetimeModel,
        reward: Reward,
        discounting_rate: float = 0.0,
        first_lifetime_model: Optional[_FrozenLike_ParametricLifetimeModel] = None,
        first_reward: Optional[Reward] = None,
    ) -> None: ...
    @property
    def discounting_rate(self) -> float: ...
    # noinspection PyUnresolvedReferences
    @discounting_rate.setter
    def discounting_rate(self, value: float) -> None: ...
    def expected_total_reward(self, tf: float, nb_steps: int) -> tuple[_Timeline, _Expected_Values]: ...
    def expected_equivalent_annual_worth(self, tf: float, nb_steps: int) -> tuple[_Timeline, _Expected_Values]: ...
    def asymptotic_expected_total_reward(self) -> _Asymptotic_Expected_Values: ...
    def asymptotic_expected_equivalent_annual_worth(self) -> _Asymptotic_Expected_Values: ...
    @override
    def sample(
        self, size: int, tf: float, t0: float = 0.0, seed: Optional[int] = None
    ) -> RenewalRewardProcessSample: ...
