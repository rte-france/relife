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
