from typing import TypeAlias

import numpy as np

__all__ = ["Seed"]

Seed: TypeAlias = int | np.random.Generator | np.random.BitGenerator | np.random.RandomState
