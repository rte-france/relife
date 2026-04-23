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