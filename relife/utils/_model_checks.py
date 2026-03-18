# pyright: basic

import numpy as np

from ._array_utils import get_args_nb_assets, reshape_1d_arg

__all__ = [
    "get_model_nb_assets",
]


def get_model_nb_assets(model):
    """
    Gets the number of assets stored by a model (frozen or not).
    """
    from relife.base import FrozenParametricModel
    from relife.lifetime_model import EquilibriumDistribution, MinimumDistribution
    from relife.lifetime_model._regression import ParametricLifetimeRegression
    from relife.stochastic_process import NonHomogeneousPoissonProcess, RenewalProcess

    if isinstance(model, EquilibriumDistribution) or isinstance(
        model, MinimumDistribution
    ):
        return get_model_nb_assets(model.baseline)

    if isinstance(model, NonHomogeneousPoissonProcess):
        return get_model_nb_assets(model.lifetime_model)

    if isinstance(model, RenewalProcess):
        lifetime_model_nb_assets = get_model_nb_assets(model.lifetime_model)
        if model.first_lifetime_model is not None:
            first_lifetime_model_nb_assets = get_model_nb_assets(
                model.first_lifetime_model
            )
            return max(lifetime_model_nb_assets, first_lifetime_model_nb_assets)
        return lifetime_model_nb_assets

    if isinstance(model, FrozenParametricModel):
        if isinstance(model._unfrozen_model, NonHomogeneousPoissonProcess):
            return get_model_nb_assets(model._unfrozen_model)
        if isinstance(model._unfrozen_model, ParametricLifetimeRegression):
            # specific covar reshape
            reshaped_args = [np.atleast_2d(model.args[0])]
            reshaped_args += [reshape_1d_arg(arg) for arg in model.args[1:]]
        else:
            reshaped_args = [reshape_1d_arg(arg) for arg in model.args]
        return get_args_nb_assets(*reshaped_args)

    return 1
