import numpy as np
import matplotlib.pyplot as plt
from optype.numpy import Array, Array1D, Array2D, ArrayND, is_array_1d
import functools
from collections.abc import Callable
from typing import Any, Literal, ParamSpec, TypeAlias, TypedDict, TypeVar
from relife.stochastic_processes import RenewalProcess
from relife.base import ParametricModel
from relife.stochastic_processes import RenewalProcess

from relife.lifetime_models._base import (
    ParametricLifetimeModel,
)

from relife.lifetime_models._conditional_models import (
    get_conditional_lifetime_model,
)

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


FT: TypeAlias = Callable[
    [ST | NumpyST | ArrayND[NumpyST]],
    np.float64 | ArrayND[np.float64],
]




def mean_age (self, tf:float, nb_steps:int,
             a0: ST | NumpyST | Array1D[NumpyST] | None = None,
             ar: ST | NumpyST | Array1D[NumpyST] | None = None,
)-> tuple[Array1D[np.float64], Array1D[np.float64] | Array2D[np.float64]] :
    r""" #La docstring pour la documentation
     mean age of the renewal process
    It gives the average age of the assets in population: math:'e'
    It is computed by solving the renewal equation
    
    .. math::
        
        e(t) = t(1-f(t)) + \int_0^t e(t-x) \mathrm{d}F(x)
    
    where :
    - math: 'F' is the cumulative distribution function of the time to failure :math: 'X'
    
    If "ar" is given, :math:'F' becomes :math:'F_{a_r}' defined by :math:'T= text{min}(X, ~a_r) \sim F_{a_r}'.
    The same applies for :math: 'X_1'. :math:'F_1' becomes :math:'F_{1_{a_r}}' defined by :math: 'T_1 = \text{min}(X_1, ~a_r) \sim F_{a_r}'.
    
    If "a0" is given, :math: 'F_1' becomes :math:'\mathbb{P}(X \leq t | ~ X > a_0)'.
    
    Parameters
    ----------
    tf : float
        The final time
    nb_steps : int
        The number of steps used to discretized the time.
    a0 : float or np.ndarray, optional
        Initial ages of the assets.
    ar : float or np.ndarray, optional
        Preventive ages of replacements.
    
    
    Returns
    -------
    out : tuple of two ndarrays
        A timeline and the corresponding values.
            
    """
    renewal_equation_solver = RenewalEquationSolver(get_conditional_lifetime_model(self.lifetime_model, ar=ar),
                                                    get_conditional_lifetime_model(self.first_lifetime_model, ar=ar, a0=a0).pdf,
    )
    return renewal_equation_solver.solve(tf, nb_steps)