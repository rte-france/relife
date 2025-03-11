from typing import Any, Self, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from relife.core.descriptors import ShapedArgs
from relife.core.likelihoods import LikelihoodFromLifetimes
from relife.core.model import LifetimeModel, ParametricModel
from relife.data import lifetime_data_factory, nhpp_lifetime_data_factory, CountData
from relife.plots import PlotNHPP, PlotConstructor
from relife.rewards import run_to_failure_rewards
from relife.types import Arg, VariadicArgs


class NonHomogeneousPoissonProcess(ParametricModel):

    model_args = ShapedArgs(astuple=True)

    def __init__(
        self,
        model: LifetimeModel[*tuple[Arg, ...]],
        model_args: tuple[Arg, ...] = (),
        *,
        nb_assets: int = 1,
        cr: Optional[NDArray[np.float64]] = None,
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

    def simulate(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> CountData:
        from relife.sampling import sample_count_data

        return sample_count_data(self, size, tf, t0=t0, maxsample=maxsample, seed=seed)

    # sample_failure_data
    def sample_failure_data(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        seed: Optional[int] = None,
        use: str = "model",
    ) -> tuple[NDArray[np.float64], ...]:
        from relife.sampling import sample_failure_data

        return sample_failure_data(self, size, tf, t0, seed, use)

    @property
    def plot(self) -> PlotConstructor:
        return PlotNHPP(self)

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
