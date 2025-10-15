def is_frozen(model):
    # local import to avoid circular import
    from relife.base import FrozenParametricModel

    return isinstance(model, FrozenParametricModel)


def is_lifetime_model(model):
    # local import to avoid circular import
    from relife.lifetime_model import ParametricLifetimeModel

    if is_frozen(model):
        return isinstance(model.unfrozen_model, ParametricLifetimeModel)
    return isinstance(model, ParametricLifetimeModel)


def is_non_homogeneous_poisson_process(model):
    # local import to avoid circular import
    from relife.stochastic_process import NonHomogeneousPoissonProcess

    if is_frozen(model):
        return isinstance(model.unfrozen_model, NonHomogeneousPoissonProcess)
    return isinstance(model, NonHomogeneousPoissonProcess)
