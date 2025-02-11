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
