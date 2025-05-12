from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray


def _time_reshape(time : NDArray[np.float64]) -> NDArray[np.float64]:
    # Check time shape
    if time.ndim > 2 or (time.ndim == 2 and time.shape[-1] not in (1, 2)):
        raise ValueError(f"Invalid time shape, got {time.shape} be time must be (m,), (m, 1) or (m,2)")
    if time.ndim < 2:
        time = time.reshape(-1, 1)  # time is (m, 1) or (m, 2)
    return time

def _event_reshape(event : Optional[NDArray[np.bool_]] = None) -> Optional[NDArray[np.bool_]]:
    if event is not None:
        if event.ndim > 2 or (event.ndim == 2 and event.shape[-1] != 1):
            raise ValueError(f"Invalid event shape, got {event.shape} be event must be (m,) or (m, 1)")
        if event.ndim < 2:
            event = event.reshape(-1, 1)
        return event

def _entry_reshape(entry: Optional[NDArray[np.float64]] = None) -> Optional[NDArray[np.float64]]:
    if entry is not None:
        if entry.ndim > 2 or (entry.ndim == 2 and entry.shape[-1] != 1):
            raise ValueError(f"Invalid entry shape, got {entry.shape} be entry must be (m,) or (m, 1)")
        if entry.ndim < 2:
            entry = entry.reshape(-1, 1)
        return entry

def _departure_reshape(departure: Optional[NDArray[np.float64]] = None) -> Optional[NDArray[np.float64]]:
    if departure is not None:
        if departure.ndim > 2 or (departure.ndim == 2 and departure.shape[-1] != 1):
            raise ValueError(f"Invalid departure shape, got {departure.shape} be departure must be (m,) or (m, 1)")
        if departure.ndim < 2:
            departure = departure.reshape(-1, 1)
        return departure

def _args_reshape(args : tuple[float | NDArray[np.float64], ...] = ()) -> tuple[NDArray[np.float64]]:
    args: list[NDArray[np.float64]] = [np.asarray(arg) for arg in args]
    for i, arg in enumerate(args):
        if arg.ndim > 2:
            raise ValueError(f"Invalid arg shape, got {arg.shape} shape at position {i}")
        if arg.ndim < 2:
            args[i] = arg.reshape(-1, 1)
    return tuple(args)


@dataclass
class LifetimeData:
    time : NDArray[np.float64]
    event: Optional[NDArray[np.bool_]] = field(default=None)
    entry: Optional[NDArray[np.float64]] = field(default=None)
    departure: Optional[NDArray[np.float64]] = field(default=None)
    args: tuple[float | NDArray[np.float64], ...] = field(default_factory=tuple)

    def __post_init__(self):
        self.time = _time_reshape(self.time)
        if self.time.shape[1] == 2 and self.event is not None:
            raise ValueError("When time is 2d, event is not necessary because time already encodes event information. Remove event")
        self.event = _event_reshape(self.event)
        self.entry = _entry_reshape(self.entry)
        self.departure = _departure_reshape(self.departure)
        self.args = _args_reshape(self.args)

        if self.event is None:
            self.event = np.ones((len(self.time), 1)).astype(np.bool_)
        if self.entry is None:
            self.entry = np.zeros((len(self.time), 1))
        if self.departure is None:
            self.departure = np.ones((len(self.time), 1)) * np.inf

        # Check sizes
        sizes = [len(x) for x in (self.time, self.event, self.entry, self.departure, *self.args) if x is not None]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"All lifetime data must have the same number of values. Fields length are different. Got {set(sizes)}")