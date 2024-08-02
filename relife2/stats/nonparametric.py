"""
This module defines fundamental types of nonparametric functions used in survival analysis

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from relife2.data import lifetime_factory_template
from relife2.utils.types import FloatArray


@dataclass
class Estimates:
    """
    BLABLABLABLA
    """

    timeline: FloatArray
    values: FloatArray
    se: Optional[FloatArray] = None

    def __post_init__(self):
        if self.se is None:
            self.se = np.zeros_like(
                self.values
            )  # garder None/Nan efaire le changement de valeur au niveau du plot

        if self.timeline.shape != self.values.shape != self.se:
            raise ValueError("Incompatible timeline, values and se in Estimates")


class NonParametricEstimators(ABC):
    """_summary_"""

    def __init__(
        self,
    ):
        self.estimations = {}

    @abstractmethod
    def estimate(
        self,
        time: ArrayLike,
        event: Optional[ArrayLike] = None,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
    ) -> None:

        """_summary_

        Returns:
            Tuple[Estimates]: description
        """
