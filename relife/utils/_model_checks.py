from ._array_api import reshape_1d_arg, get_args_nb_assets
import numpy as np


def is_frozen(model):
    """
    Checks if model is frozen.
    """
    from relife.base import FrozenParametricModel

    return isinstance(model, FrozenParametricModel)


def is_lifetime_model(model):
    """
    Checks if model is a lifetime model.
    """
    # local import to avoid circular import
    from relife.lifetime_model import ParametricLifetimeModel

    if is_frozen(model):
        return isinstance(model.unfrozen_model, ParametricLifetimeModel)
    return isinstance(model, ParametricLifetimeModel)


def is_non_homogeneous_poisson_process(model):
    """
    Checks if model is a non-homogeneous Poisson process.
    """
    # local import to avoid circular import
    from relife.stochastic_process import NonHomogeneousPoissonProcess

    if is_frozen(model):
        return isinstance(model.unfrozen_model, NonHomogeneousPoissonProcess)
    return isinstance(model, NonHomogeneousPoissonProcess)


def get_model_nb_assets(model):
    """
    Gets the number of assets stored by a model (frozen or not).
    """
    from relife.base import FrozenParametricModel
    from relife.lifetime_model import EquilibriumDistribution, MinimumDistribution, LifetimeRegression
    from relife.stochastic_process import NonHomogeneousPoissonProcess
    from relife.stochastic_process import RenewalProcess

    if isinstance(model, EquilibriumDistribution) or isinstance(model, MinimumDistribution):
        return get_model_nb_assets(model.baseline)

    if isinstance(model, NonHomogeneousPoissonProcess):
        return get_model_nb_assets(model.lifetime_model)

    if isinstance(model, RenewalProcess):
        lifetime_model_nb_assets = get_model_nb_assets(model.lifetime_model)
        if model.first_lifetime_model is not None:
            first_lifetime_model_nb_assets = get_model_nb_assets(model.first_lifetime_model)
            return max(lifetime_model_nb_assets, first_lifetime_model_nb_assets)
        return lifetime_model_nb_assets

    if isinstance(model, FrozenParametricModel):
        if isinstance(model.unfrozen_model, NonHomogeneousPoissonProcess):
            return get_model_nb_assets(model.unfrozen_model)
        if isinstance(model.unfrozen_model, LifetimeRegression):
            # specific covar reshape
            reshaped_args = [np.atleast_2d(model.args[0])]
            reshaped_args += [reshape_1d_arg(arg) for arg in model.args[1:]]
        else:
            reshaped_args = [reshape_1d_arg(arg) for arg in model.args]
        return get_args_nb_assets(*reshaped_args)

    return 1