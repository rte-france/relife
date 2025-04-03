from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    Self,
    Sequence,
    TypeVarTuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from relife._plots import PlotNHPP
from relife.data import nhpp_data_factory
from relife.likelihood.mle import maximum_likelihood_estimation
from relife.model import (
    FrozenNonHomogeneousPoissonProcess,
    ParametricLifetimeModel,
    ParametricModel,
)
from relife.sample import SampleFailureDataMixin, SampleMixin

if TYPE_CHECKING:
    from relife.likelihood.mle import FittingResults

Args = TypeVarTuple("Args")


class NonHomogeneousPoissonProcess(
    ParametricModel, SampleMixin[*Args], SampleFailureDataMixin[*Args], Generic[*Args]
):

    fitting_results: Optional[FittingResults]

    def __init__(
        self,
        baseline: ParametricLifetimeModel[*Args],
    ):
        super().__init__()
        self.compose_with(baseline=baseline)
        self.baseline = baseline

    def intensity(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return self.baseline.hf(time, *args)

    def cumulative_intensity(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return self.baseline.chf(time, *args)

    # def sample(
    #     self,
    #     size: int,
    #     tf: float,
    #     /,
    #     *args: *Args,
    #     t0: float = 0.0,
    #     maxsample: int = 1e5,
    #     seed: Optional[int] = None,
    # ) -> CountData:
    #     from relife.sample import sample_count_data
    #
    #     return sample_count_data(
    #         self.baseline.freeze(*args),
    #         size,
    #         tf,
    #         t0=t0,
    #         maxsample=maxsample,
    #         seed=seed,
    #     )
    #
    # def failure_data_sample(
    #     self,
    #     size: int,
    #     tf: float,
    #     /,
    #     *args: *Args,
    #     t0: float = 0.0,
    #     maxsample: int = 1e5,
    #     seed: Optional[int] = None,
    # ) -> tuple[NDArray[np.float64], ...]:
    #     from relife.sample import failure_data_sample
    #
    #     return failure_data_sample(
    #         self.baseline.freeze(*args),
    #         size,
    #         tf,
    #         t0,
    #         maxsample=maxsample,
    #         seed=seed,
    #         use="model",
    #     )

    def freeze(self, *args: *Args) -> FrozenNonHomogeneousPoissonProcess:
        return FrozenNonHomogeneousPoissonProcess(self, *args)

    @property
    def plot(self) -> PlotNHPP:
        return PlotNHPP(self)

    def fit(
        self,
        events_assets_ids: Union[Sequence[str], NDArray[np.int64]],
        events_ages: NDArray[np.float64],
        /,
        *args: *Args,
        assets_ids: Optional[Union[Sequence[str], NDArray[np.int64]]] = None,
        first_ages: Optional[NDArray[np.float64]] = None,
        last_ages: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> Self:

        nhpp_data = nhpp_data_factory(
            events_assets_ids,
            events_ages,
            *args,
            assets_ids=assets_ids,
            first_ages=first_ages,
            last_ages=last_ages,
        )
        optimized_model = maximum_likelihood_estimation(self, nhpp_data, **kwargs)
        return optimized_model
