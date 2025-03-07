from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from relife.core.descriptors import ShapedArgs
from relife.core.likelihoods import LikelihoodFromLifetimes
from relife.core.model import LifetimeModel, ParametricModel
from relife.data import lifetime_data_factory, nhpp_lifetime_data_factory
from relife.types import Arg, VariadicArgs


class NonHomogeneousPoissonProcess(ParametricModel):

    model_args = ShapedArgs(astuple=True)

    def __init__(
        self,
        model: LifetimeModel[*tuple[Arg, ...]],
        model_args: tuple[Arg, ...] = (),
        *,
        nb_assets: int = 1,
    ):
        super().__init__()
        self.nb_assets = nb_assets
        self.compose_with(model=model)
        self.model_args = model_args

    def intensity(self, time: np.ndarray, *args: *tuple[Arg, ...]) -> np.ndarray:
        return self.model.hf(time, *args)

    def cumulative_intensity(
        self, time: np.ndarray, *args: *tuple[Arg, ...]
    ) -> np.ndarray:
        return self.model.chf(time, *args)

    def fit(
        self,
        a0: NDArray[np.float64],
        af: NDArray[np.float64],
        ages: NDArray[np.float64],
        assets: NDArray[np.int64],
        model_args: tuple[*VariadicArgs] = (),
        **kwargs: Any,
    ) -> Self:

        lifetime_data = lifetime_data_factory(
            *nhpp_lifetime_data_factory(a0, af, ages, assets)
        )

        optimized_model = self.model.copy()
        optimized_model.init_params(lifetime_data, *model_args)
        # or just optimized_model.init_params(observed_lifetimes, *model_args)

        likelihood = LikelihoodFromLifetimes(
            optimized_model, lifetime_data, model_args=model_args
        )

        minimize_kwargs = {
            "method": kwargs.get("method", "Nelder-Mead"),  # Nelder-Mead better here
            "constraints": kwargs.get("constraints", ()),
            "tol": kwargs.get("tol", None),
            "callback": kwargs.get("callback", None),
            "options": kwargs.get("options", None),
            "bounds": kwargs.get("bounds", optimized_model.params_bounds),
            "x0": kwargs.get("x0", optimized_model.params),
        }

        optimizer = minimize(
            likelihood.negative_log,
            minimize_kwargs.pop("x0"),
            # jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )
        optimized_model.params = optimizer.x

        self.model.init_params(lifetime_data, *model_args)
        self.model.params = optimized_model.params

        return self
