from typing import TypeAlias, TypeVarTuple

import numpy as np
from optype.numpy import ArrayND

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint

Ts = TypeVarTuple("Ts")
VT: TypeAlias = ST | NumpyST | ArrayND[NumpyST]
