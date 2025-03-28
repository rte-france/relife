import functools

ERROR_MESSAGE_TEMPLATE = "{attr_name} is not set. If fit exists, you may need to fit the object first or instanciate the object with {attr_name}"


def choose(*attribute_names: str):
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            for attr_name in attribute_names:
                attr_value = getattr(self, attr_name)
                arg_value = kwargs.pop(attr_name, None)
                if attr_value is None and arg_value is None:
                    raise ValueError(ERROR_MESSAGE_TEMPLATE.format(attr_name=attr_name))
                if attr_value is not None and arg_value is not None:
                    raise ValueError
                elif attr_value is not None and arg_value is None:
                    kwargs[attr_name] = attr_value
                else:
                    kwargs[attr_name] = arg_value

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
