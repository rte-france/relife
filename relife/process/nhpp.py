from typing import Optional, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from relife.core.likelihoods import LikelihoodFromLifetimes
from relife.core.model import LifetimeModel, ParametricModel
from relife.data import lifetime_data_factory, NHPPData, nhpp_lifetime_data_factory
from relife.generator import nhpp_generator
from relife.types import ModelArgs, VariadicArgs


class NHPP(ParametricModel):

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        model_args: ModelArgs = (),
    ):
        super().__init__()
        self.compose_with(model=model)
        self.model_args = model_args

    def intensity(self, time: np.ndarray, *args: *ModelArgs) -> np.ndarray:
        return self.model.hf(time, *args)

    def cumulative_intensity(self, time: np.ndarray, *args: *ModelArgs) -> np.ndarray:
        return self.model.chf(time, *args)

    def sample(
        self, nb_samples, nb_assets, end_time: int, seed: Optional[int] = None
    ) -> NHPPData:
        durations = np.array([], dtype=np.float64)
        ages = np.array([], dtype=np.float64)
        samples = np.array([], dtype=np.int64)
        assets = np.array([], dtype=np.int64)
        order = np.array([], dtype=np.int64)

        for i, (_durations, _ages, still_valid) in enumerate(
            nhpp_generator(
                self.model,
                nb_samples,
                nb_assets,
                end_time,
                model_args=self.model_args,
                seed=seed,
            )
        ):
            durations = np.concatenate((durations, _durations[still_valid]))
            ages = np.concatenate((ages, _ages[still_valid]))
            order = np.concatenate((order, np.ones_like(_ages[still_valid]) * i))
            _assets, _samples = np.where(still_valid)
            samples = np.concatenate((samples, _samples))
            assets = np.concatenate((assets, _assets))

        # note that ages == event_times
        return NHPPData(samples, assets, order, ages, durations)

    def fit(
        self,
        a0: NDArray[np.float64],
        af: NDArray[np.float64],
        ages: NDArray[np.float64],
        assets: NDArray[np.int64],
        model_args: tuple[*VariadicArgs] = (),
        inplace: bool = True,
        **kwargs: Any,
    ) -> NDArray[np.float64]:

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
            "method": kwargs.get("method", "L-BFGS-B"),
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
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )

        if inplace:
            self.model.init_params(lifetime_data, *model_args)
            # or just self.init_params(observed_lifetimes, *model_args)
            self.model.params = likelihood.params

        return optimizer.x
