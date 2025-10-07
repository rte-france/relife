from relife.base import FrozenParametricModel


def is_frozen(model):
    return isinstance(model, FrozenParametricModel)
