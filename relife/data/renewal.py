from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .counting import CountData


@dataclass
class RenewalData(CountData):
    durations: NDArray[np.float64] = field(repr=False)

    # def iter(self, sample_id: Optional[int] = None):
    #     if sample_id is None:
    #         return CountDataIterable(self, ("event_times", "lifetimes", "events"))
    #     else:
    #         if sample_id not in self.samples_unique_ids:
    #             raise ValueError(f"{sample_id} is not part of samples index")
    #         return filterfalse(
    #             lambda x: x[0] != sample_id,
    #             CountDataIterable(self, ("event_times", "lifetimes", "events")),
    #         )


@dataclass
class RenewalRewardData(RenewalData):
    rewards: NDArray[np.float64] = field(repr=False)

    def cum_total_rewards(
        self, sample: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind = self.samples_ids == sample
        s = np.argsort(self.timeline[ind])
        times = np.insert(self.timeline[ind][s], 0, 0)
        z = np.insert(self.rewards[ind][s].cumsum(), 0, 0)
        return times, z

    def mean_total_reward(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = np.argsort(self.timeline)
        times = np.insert(self.timeline[s], 0, 0)
        z = np.insert(self.rewards[s].cumsum(), 0, 0) / self.nb_samples
        return times, z
