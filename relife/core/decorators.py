import functools

ERROR_MESSAGE_TEMPLATE = (
    "{attr_name} is not set. If fit exists, you may need to fit the object first"
)


def require_attributes(*attribute_names: str):
    """
    A decorator to ensure that specified attributes are set on the instance
    before executing the decorated method.

    Parameters:
    attribute_names: A list of attribute names (strings) that must not be None.

    Raises:
        ValueError: If any of the specified attributes is None.
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            for attr_name in attribute_names:
                attr_value = getattr(self, attr_name)
                if attr_value is None:
                    raise ValueError(ERROR_MESSAGE_TEMPLATE.format(attr_name=attr_name))
            return method(self, *args, **kwargs)

        return wrapper

    return decorator


def isbroadcastable(argname: str):
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, x):
            if x.ndim == 2:
                if x.shape[0] != 1 and x.shape[0] != self.nb_assets:
                    raise ValueError(
                        f"Inconsistent {argname} shape. Got {self.nb_assets} nb of assets but got {x.shape} {argname} shape"
                    )
            return method(self, x)

        return wrapper

    return decorator
