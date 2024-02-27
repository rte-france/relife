from dataclasses import InitVar, asdict, dataclass, field

import numpy as np
from scipy.optimize import OptimizeResult


@dataclass
class FittingResult:
    """Fitting results of the parametric model."""

    opt: OptimizeResult = field(
        repr=False
    )  #: Optimization result (see scipy.optimize.OptimizeResult doc).
    jac: np.ndarray = field(
        repr=False
    )  #: Jacobian of the negative log-likelihood with the lifetime data.
    var: np.ndarray = field(
        repr=False
    )  #: Covariance matrix (computed as the inverse of the Hessian matrix)
    se: np.ndarray = field(
        init=False, repr=False
    )  #: Standard error, square root of the diagonal of the covariance matrix.
    nb_samples: int  #: Number of observations (samples).
    nb_params: int = field(init=False)  #: Number of parameters.
    AIC: float = field(init=False)  #: Akaike Information Criterion.
    AICc: float = field(
        init=False
    )  #: Akaike Information Criterion with a correction for small sample sizes.
    BIC: float = field(init=False)  #: Bayesian Information Criterion.

    def __post_init__(self):
        self.se = np.sqrt(np.diag(self.var))
        self.nb_params = self.opt.x.size
        self.AIC = 2 * self.nb_params + 2 * self.opt.fun
        self.AICc = self.AIC + 2 * self.nb_params * (self.nb_params + 1) / (
            self.nb_samples - self.nb_params - 1
        )
        self.BIC = np.log(self.nb_samples) * self.nb_params + 2 * self.opt.fun

    def standard_error(self, jac_f: np.ndarray) -> np.ndarray:
        """Standard error estimation function.

        Parameter
        ----------
        jac_f : 1D array
            The Jacobian of a function f with respect to params.

        Returns
        -------
        1D array
            Standard error for f(params).

        References
        ----------
        .. [1] Meeker, W. Q., Escobar, L. A., & Pascual, F. G. (2022).
            Statistical methods for reliability data. John Wiley & Sons.
        """
        # [1] equation B.10 in Appendix
        return np.sqrt(np.einsum("ni,ij,nj->n", jac_f, self.var, jac_f))

    def asdict(self) -> dict:
        """converts FittingResult into a dictionary.

        Returns
        -------
        dict
            Returns the fitting result as a dictionary.
        """
        return asdict(self)


@dataclass
class Parameter:
    nb_params: InitVar[int] = None
    param_names: InitVar[list] = None

    def __post_init__(self, nb_params, param_names):
        if nb_params is not None and param_names is not None:
            if {type(name) for name in param_names} != {str}:
                raise ValueError("param_names must be string")
            if len(param_names) != nb_params:
                raise ValueError(
                    "param_names must have same length as nb_params"
                )
            self.nb_params = nb_params
            self.param_names = param_names
        elif nb_params is not None and param_names is None:
            self.nb_params = nb_params
            self.param_names = [f"param_{i}" for i in range(nb_params)]
        elif nb_params is None and param_names is not None:
            if {type(name) for name in param_names} != {str}:
                raise ValueError("param_names must be string")
            self.nb_params = len(param_names)
            self.param_names = param_names
        else:
            raise ValueError(
                """
            Parameter expects at least nb_params or param_names
            """
            )

        self.values = np.random.rand(self.nb_params)
        self.fitting_params = None
        self.params_index = {
            name: i for i, name in enumerate(self.param_names)
        }

    def __len__(self):
        return self.nb_params

    def __getitem__(self, i):
        return self.values[i]

    def __getattr__(self, attr: str):
        """
        called if attr is not found in attributes of the class
        (different from __getattribute__)
        """
        if attr in self.params_index:
            return self.values[self.params_index[attr]]
        else:
            raise AttributeError(
                f"""
                Parameter has no attr called {attr}
                """
            )

    def __str__(self):
        print(self.nb_params, self.param_names)
        class_name = type(self).__name__
        res = [
            f"{name} = {getattr(self, name)} \n" for name in self.param_names
        ]
        res = ", ".join(res)
        return f"{class_name}\n{res}"


class ModelParameters:
    def __init__(self, *params: Parameter, names: list = None):
        pass
