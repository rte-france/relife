from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Policy(Protocol):
    """
    Policy structural type
    """

    # warning: tf > 0, period > 0, dt is deduced from period and is < 0.5
    def expected_total_cost(
        self, timeline: NDArray[np.float64]  # tf: float, period:float=1
    ) -> NDArray[np.float64]:
        """The expected total discounted cost.

        It is computed bu solving the renewal equation.

        Parameters
        ----------
        timeline : ndarray
            Timeline of points where the function is evaluated

        Returns
        -------
        ndarray
            Expected values along the timeline
        """

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The expected equivalent annual cost.


        Parameters
        ----------
        timeline : ndarray
            Timeline of points where the function is evaluated

        Returns
        -------
        ndarray
            Expected values along the timeline
        """
