from typing import Optional, Any, Self

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
        if self.cumulative_intensity(end_time) * nb_samples * nb_assets > 1e6:
            raise ValueError("Exceeded number of sample allowed")
        durations = np.array([], dtype=np.float64)
        ages = np.array([], dtype=np.float64)
        samples_ids = np.array([], dtype=np.int64)
        assets_ids = np.array([], dtype=np.int64)

        for _durations, _ages, still_valid in nhpp_generator(
            self.model,
            nb_samples,
            nb_assets,
            end_time,
            model_args=self.model_args,
            seed=seed,
        ):
            durations = np.concatenate((durations, _durations[still_valid]))
            ages = np.concatenate((ages, _ages[still_valid]))
            _assets, _samples = np.where(still_valid)
            samples_ids = np.concatenate((samples_ids, _samples))
            assets_ids = np.concatenate((assets_ids, _assets))

        # note that ages == event_times
        return NHPPData(samples_ids, assets_ids, ages, durations)

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
