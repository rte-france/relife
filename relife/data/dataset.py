from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_power_transformer() -> NDArray[np.float64]:
    """BLABLA"""
    data = np.loadtxt(
        Path(Path(__file__).parents[0], "csv/power_transformer.csv"),
        delimiter=",",
        skiprows=1,
        unpack=True,
        dtype=np.dtype([("time", np.float64), ("event", np.bool_), ("entry", np.float64)]),
    )

    return data


def load_insulator_string() -> NDArray[np.float64]:
    """BLABLA"""

    data = np.loadtxt(
        Path(Path(__file__).parents[0], "csv/insulator_string.csv"),
        delimiter=",",
        skiprows=1,
        unpack=True,
        dtype=np.dtype(
            [
                ("time", np.float64),
                ("event", np.bool_),
                ("entry", np.float64),
                ("pHCl", np.float64),
                ("pH2SO4", np.float64),
                ("HNO3", np.float64),
            ]
        ),
    )
    return data


def load_circuit_breaker() -> NDArray[np.float64]:
    """BLABLA"""

    data = np.loadtxt(
        Path(Path(__file__).parents[0], "csv/circuit_breaker.csv"),
        delimiter=",",
        skiprows=1,
        unpack=True,
        dtype=np.dtype([("time", np.float64), ("event", np.bool_), ("entry", np.float64)]),
    )
    return data


def load_input_turnbull() -> NDArray[np.float64]:
    """_summary_

    Returns:
        np.ndarray: _description_
    """
    data = np.loadtxt(
        Path(Path(__file__).parents[0], "csv/input_turnbull.csv"),
        delimiter=",",
        skiprows=1,
        unpack=True,
    )
    return data
