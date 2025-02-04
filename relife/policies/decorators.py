import functools


def ifset(*param_names: str):
    """
    simple decorator to check if some params are set before executed one method
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            for name in param_names:
                if getattr(self, name) is None:
                    raise ValueError(
                        f"{name} is not set. If fit exists, you may need to fit the policy first"
                    )
            return method(self, *args, **kwargs)

        return wrapper

    return decorator
