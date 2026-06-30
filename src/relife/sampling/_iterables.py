from typing_extensions import override

from relife.stochastic_processes import (
    Kijima1Process,
    Kijima2Process,
    NonHomogeneousPoissonProcess,
    RenewalProcess,
    RenewalRewardProcess,
)

from ._base import StochasticDataIterable, StochasticDataIterator
from ._iterators import (
    Kijima1ProcessIterator,
    Kijima2ProcessIterator,
    NonHomogeneousPoissonProcessIterator,
    RenewalProcessIterator,
    RenewalRewardProcessIterator,
)


class RenewalProcessIterable(StochasticDataIterable[RenewalProcess]):
    @override
    def __iter__(self) -> StochasticDataIterator[RenewalProcess]:
        return RenewalProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            self.a0,
            self.ar,
            self.seed,
        )


class RenewalRewardProcessIterable(StochasticDataIterable[RenewalRewardProcess]):
    @override
    def __iter__(self) -> StochasticDataIterator[RenewalRewardProcess]:
        return RenewalRewardProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            self.a0,
            self.ar,
            self.seed,
        )


class NonHomogeneousPoissonProcessIterable(
    StochasticDataIterable[NonHomogeneousPoissonProcess[()]]
):
    @override
    def __iter__(self) -> StochasticDataIterator[NonHomogeneousPoissonProcess[()]]:
        return NonHomogeneousPoissonProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            self.a0,
            self.ar,
            self.seed,
        )


class Kijima1ProcessIterable(StochasticDataIterable[Kijima1Process[()]]):
    @override
    def __iter__(self) -> StochasticDataIterator[Kijima1Process[()]]:
        return Kijima1ProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            self.a0,
            self.ar,
            self.seed,
        )


class Kijima2ProcessIterable(StochasticDataIterable[Kijima2Process[()]]):
    @override
    def __iter__(self) -> StochasticDataIterator[Kijima2Process[()]]:
        return Kijima2ProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            self.a0,
            self.ar,
            self.seed,
        )
