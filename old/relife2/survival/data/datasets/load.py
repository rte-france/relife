from pathlib import Path
from typing import Tuple

import numpy as np


def load_power_transformer() -> Tuple[np.ndarray]:
    """Load and return the power transformer lifetime data.

    The data were simulated from a real estimate:

    - `time`: time-to-event or durations in years,
    - `event`: if a failure occurs during the observation period,
    - `entry`: age of the power transformers in years at the beginning of the
      observation period.

    Returns
    -------
    LifetimeData
        The lifetime data as a dataclass instance.

    Examples
    --------

    .. code-block::

        from relife.datasets import load_power_transformer
        time, event, entry = load_power_transformer().astuple()
    """
    data = np.loadtxt(
        Path(Path(__file__).parent, "power_transformer.csv"),
        delimiter=",",
        skiprows=1,
        unpack=True,
    )

    return data
