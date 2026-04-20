import numpy as np

from relife.typing._scalars import AnyFloat, NumpyFloat


def apply_bias(
    t: AnyFloat,
    delay: NumpyFloat | None = None,
    censor_value: NumpyFloat | None = None,
    delay_applied_to_censor=False,
):
    if delay_applied_to_censor:
        return np.minimum(t, (censor_value or np.inf) - (delay or 0))
    return np.minimum(t + (delay or 0), (censor_value or np.inf))