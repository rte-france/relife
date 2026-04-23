import numpy as np

from relife.typing._scalars import AnyFloat, NumpyFloat


def apply_bias(
    t: AnyFloat,
    delay: NumpyFloat | None = None,
    censor_value: NumpyFloat | None = None,
    delay_applied_to_censor=False,
):
    if censor_value is None:
        censor_value = np.inf
    if delay is None:
        delay = 0
    if delay_applied_to_censor:
        return np.minimum(t, censor_value - delay)
    return np.minimum(t + delay,censor_value)



def with_reshape_a0_ar(func):
    sig = inspect.signature(func)
    params = sig.parameters

    def wrapper(*args, **kwargs):
        if "a0" in params and kwargs.get("a0") is not None:
            kwargs["a0"] = reshape_1d_arg(kwargs["a0"])

        if "ar" in params and kwargs.get("ar") is not None:
            kwargs["ar"] = reshape_1d_arg(kwargs["ar"])

        return func(*args, **kwargs)

    return wrapper