from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from relife2.survival.data.lifetimes import LifetimeData, ObservedLifetimes, Truncations
from relife2.survival.data.tools import lifetimes_compatibility, array_factory

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class LifetimeDataFactory(ABC):
    """
    Factory method of ObservedLifetimes and Truncations
    """

    def __init__(
        self,
        time: FloatArray,
        entry: Optional[FloatArray] = None,
        departure: Optional[FloatArray] = None,
        lc_indicators: Optional[BoolArray] = None,
        rc_indicators: Optional[BoolArray] = None,
    ):

        if entry is None:
            entry = np.zeros((len(time), 1))

        if departure is None:
            departure = np.ones((len(time), 1)) * np.inf

        if lc_indicators is None:
            lc_indicators = np.zeros((len(time), 1)).astype(np.bool_)

        if rc_indicators is None:
            rc_indicators = np.zeros((len(time), 1)).astype(np.bool_)

        self.time = time
        self.entry = entry
        self.departure = departure
        self.lc_indicators = lc_indicators
        self.rc_indicators = rc_indicators

        if np.any(np.logical_and(self.lc_indicators, self.rc_indicators)) is True:
            raise ValueError(
                """
                lc_indicators and rc_indicators can't be true at the same index
                """
            )

        for values in (
            self.entry,
            self.departure,
            self.lc_indicators,
            self.rc_indicators,
        ):
            if values.shape != (len(self.time), 1):
                raise ValueError("invalid argument shape")

    @abstractmethod
    def get_complete(self) -> LifetimeData:
        """
        Returns:
            LifetimeData: object containing complete lifetime values and index
        """

    @abstractmethod
    def get_left_censorships(self) -> LifetimeData:
        """
        Returns:
            LifetimeData: object containing left censorhips values and index
        """

    @abstractmethod
    def get_right_censorships(self) -> LifetimeData:
        """
        Returns:
            LifetimeData: object containing right censorhips values and index
        """

    @abstractmethod
    def get_interval_censorships(self) -> LifetimeData:
        """
        Returns:
            LifetimeData: object containing interval censorhips valuess and index
        """

    @abstractmethod
    def get_left_truncations(self) -> LifetimeData:
        """
        Returns:
            LifetimeData: object containing left truncations values and index
        """

    @abstractmethod
    def get_right_truncations(self) -> LifetimeData:
        """
        Returns:
            LifetimeData: object containing right truncations values and index
        """

    def __call__(
        self,
    ) -> tuple[
        ObservedLifetimes,
        Truncations,
    ]:
        observed_lifetimes = ObservedLifetimes(
            self.get_complete(),
            self.get_left_censorships(),
            self.get_right_censorships(),
            self.get_interval_censorships(),
        )
        truncations = Truncations(
            self.get_left_truncations(),
            self.get_right_truncations(),
        )

        try:
            lifetimes_compatibility(observed_lifetimes, truncations)
        except Exception as error:
            raise ValueError("Incorrect input lifetimes") from error
        return observed_lifetimes, truncations


class LifetimeDataFactoryFrom1D(LifetimeDataFactory):
    """
    Concrete implementation of LifetimeDataFactory for 1D encoding
    """

    def get_complete(self) -> LifetimeData:
        self.lc_indicators: BoolArray
        self.rc_indicators: BoolArray
        index = np.where(np.logical_and(~self.lc_indicators, ~self.rc_indicators))[0]
        values = self.time[index]
        return LifetimeData(values, index)

    def get_left_censorships(self) -> LifetimeData:
        index = np.where(self.lc_indicators)[0]
        values = self.time[index]
        return LifetimeData(values, index)

    def get_right_censorships(self) -> LifetimeData:
        index = np.where(self.rc_indicators)[0]
        values = self.time[index]
        return LifetimeData(values, index)

    def get_interval_censorships(self) -> LifetimeData:
        return LifetimeData(np.empty((0, 2)), np.empty((0,), dtype=np.int64))

    def get_left_truncations(self) -> LifetimeData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        return LifetimeData(values, index)

    def get_right_truncations(self) -> LifetimeData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return LifetimeData(values, index)


class LifetimeDataFactoryFrom2D(LifetimeDataFactory):
    """
    Concrete implementation of LifetimeDataFactory for 2D encoding
    """

    def get_complete(self) -> LifetimeData:
        index = np.where(self.time[:, 0] == self.time[:, 1])[0]
        values = self.time[index, 0, None]
        return LifetimeData(values, index)

    def get_left_censorships(
        self,
    ) -> LifetimeData:
        index = np.where(self.time[:, 0] == 0.0)[0]
        values = self.time[index, 1, None]
        return LifetimeData(values, index)

    def get_right_censorships(
        self,
    ) -> LifetimeData:
        index = np.where(self.time[:, 1] == np.inf)[0]
        values = self.time[index, 0, None]
        return LifetimeData(values, index)

    def get_interval_censorships(self) -> LifetimeData:
        index = np.where(
            np.logical_and(
                np.logical_and(
                    self.time[:, 0] > 0,
                    self.time[:, 1] < np.inf,
                ),
                np.not_equal(self.time[:, 0], self.time[:, 1]),
            )
        )[0]
        values = self.time[index]
        lifetimes = LifetimeData(values, index)
        if len(lifetimes) != 0:
            if np.any(lifetimes.values[:, 0] >= lifetimes.values[:, 1]):
                raise ValueError(
                    "Interval censorships lower bounds can't be higher or equal to its upper bounds"
                )
        return lifetimes

    def get_left_truncations(self) -> LifetimeData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        return LifetimeData(values, index)

    def get_right_truncations(self) -> LifetimeData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return LifetimeData(values, index)


def lifetime_factory_template(
    time: ArrayLike,
    entry: Optional[ArrayLike] = None,
    departure: Optional[ArrayLike] = None,
    lc_indicators: Optional[ArrayLike] = None,
    rc_indicators: Optional[ArrayLike] = None,
) -> Tuple[ObservedLifetimes, Truncations]:
    """
    Args:
        time ():
        entry ():
        departure ():
        lc_indicators ():
        rc_indicators ():

    Returns:

    """

    time = array_factory(time)

    if entry is not None:
        entry = array_factory(entry)

    if departure is not None:
        departure = array_factory(departure)

    if lc_indicators is not None:
        lc_indicators = array_factory(lc_indicators).astype(np.bool_)

    if rc_indicators is not None:
        rc_indicators = array_factory(rc_indicators).astype(np.bool_)

    factory: LifetimeDataFactory
    if time.shape[-1] == 1:
        factory = LifetimeDataFactoryFrom1D(
            time,
            entry,
            departure,
            lc_indicators,
            rc_indicators,
        )
    else:
        factory = LifetimeDataFactoryFrom2D(
            time,
            entry,
            departure,
            lc_indicators,
            rc_indicators,
        )
    return factory()
