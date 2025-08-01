from typing import Optional, TypeAlias, TypedDict, TypeVar

import numpy as np
from numpy.typing import NDArray as NDArray
from typing_extensions import override

from relife._typing import _ParametricLifetimeModel
from relife.economic import ExponentialDiscounting, Reward

from ._sample import RenewalProcessSample, RenewalRewardProcessSample
from .base import StochasticProcess as StochasticProcess

_N = TypeVar("_N", bound=int)
_M: TypeAlias = int
_TL: TypeAlias = np.ndarray[tuple[_N], np.dtype[np.float64]] | np.ndarray[tuple[_M, _N], np.dtype[np.float64]]
_C: TypeAlias = np.ndarray[tuple[_N], np.dtype[np.float64]] | np.ndarray[tuple[_M, _N], np.dtype[np.float64]]
_A_C: TypeAlias = np.float64 | np.ndarray[tuple[_M], np.dtype[np.float64]]

class LifetimeFitArg(TypedDict):
    time: NDArray[np.float64]
    event: NDArray[np.bool_]
    entry: NDArray[np.float64]
    args: tuple[NDArray[np.float64], ...]

class RenewalProcess(StochasticProcess[()]):
    lifetime_model: _ParametricLifetimeModel[()]
    first_lifetime_model: Optional[_ParametricLifetimeModel[()]]
    def __init__(
        self,
        lifetime_model: _ParametricLifetimeModel[()],
        first_lifetime_model: Optional[_ParametricLifetimeModel[()]] = None,
    ) -> None: ...
    def _make_timeline(self, tf: float, nb_steps: _N) -> _TL[_N]: ...
    def renewal_function(self, tf: float, nb_steps: _N) -> tuple[_TL[_N], _C[_N]]: ...
    def renewal_density(self, tf: float, nb_steps: _N) -> tuple[_TL[_N], _C[_N]]: ...
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
        lifetime_model: _ParametricLifetimeModel[()],
        reward: Reward,
        discounting_rate: float = 0.0,
        first_lifetime_model: Optional[_ParametricLifetimeModel[()]] = None,
        first_reward: Optional[Reward] = None,
    ) -> None: ...
    @property
    def discounting_rate(self) -> float: ...
    # noinspection PyUnresolvedReferences
    @discounting_rate.setter
    def discounting_rate(self, value: float) -> None: ...
    @override
    def _make_timeline(self, tf: float, nb_steps: _N) -> _TL[_N]: ...
    def expected_total_reward(self, tf: float, nb_steps: _N) -> tuple[_TL[_N], _C[_N]]: ...
    def expected_equivalent_annual_worth(self, tf: float, nb_steps: int) -> tuple[_TL[_N], _C[_N]]: ...
    def asymptotic_expected_total_reward(self) -> _A_C: ...
    def asymptotic_expected_equivalent_annual_worth(self) -> _A_C: ...
    @override
    def sample(
        self, size: int, tf: float, t0: float = 0.0, seed: Optional[int] = None
    ) -> RenewalRewardProcessSample: ...
