"""
This module defines utils functions 
Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from functools import wraps

import numpy as np

from relife2.stats.functions import ParametricFunctions


def are_params_set(functions: ParametricFunctions):
    """
    Args:
        functions ():

    Returns:
    """
    if None in functions.all_params.values():
        params_to_set = " ".join(
            [name for name, value in functions.all_params.items() if value is None]
        )
        raise ValueError(
            f"Params {params_to_set} unset. Please set them first or fit the model."
        )


def squeeze(method):
    """
    Args:
        method ():

    Returns:
    """

    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        method_output = method(self, *method_args, **method_kwargs)
        return np.squeeze(method_output)[()]

    return _impl
