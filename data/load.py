from pathlib import Path

import numpy as np


def load_power_transformer() -> np.ndarray:
    """BLABLA"""
    data = np.loadtxt(
        Path(Path(__file__).parent, "power_transformer.csv"),
        delimiter=",",
        skiprows=1,
        unpack=True,
    )

    return data
