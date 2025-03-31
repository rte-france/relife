from __future__ import annotations

import functools

from relife.economic.costs import _reshape_like


def get_if_none(*args_names: str):
    """
    Decorators that get the attribute value if argument value is None
    Reshape depending on number of assets.
    If both are None, an error is raised.
    Priority is always given to the attribute value

    Parameters
    ----------
    args_names

    Returns
    -------

    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            for name in args_names:
                attr_value = getattr(self, name)
                arg_value = kwargs.pop(name, None)
                if attr_value is None and arg_value is None:
                    raise ValueError(
                        f"{name} is not set. If fit exists, you may need to fit the object first or instanciate the object with {name}"
                    )
                elif attr_value is not None and arg_value is not None:
                    # priority on arg
                    kwargs[name] = _reshape_like(arg_value, self.nb_assets)
                elif attr_value is not None and arg_value is None:
                    kwargs[name] = _reshape_like(attr_value, self.nb_assets)
                else:
                    kwargs[name] = _reshape_like(arg_value, self.nb_assets)

            return method(self, *args, **kwargs)

        return wrapper

    return decorator
