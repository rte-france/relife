"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import warnings
from typing import Union

import numpy as np

from relife2.survival.data import Lifetimes, ObservedLifetimes, Truncations
from relife2.survival.types import FloatArray, Likelihood, ParametricHazard


class LikelihoodFromLifetimes(Likelihood):
    """
    BLABLABLA
    """

    def __init__(
        self,
        functions: ParametricHazard,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
        **kwdata: FloatArray,
    ):
        super().__init__(functions)
        self.observed_lifetimes = observed_lifetimes
        self.truncations = truncations
        self.kwdata = kwdata
        for extra_arg in self.functions.extra_arguments:
            if extra_arg not in self.kwdata:
                class_name = self.functions.__class__.__name__
                raise ValueError(
                    f"kwdata must contain values of {extra_arg} to work with {class_name}"
                )
        for name in kwdata:
            if not hasattr(self.functions, name):
                class_name = self.functions.__class__.__name__
                raise AttributeError(
                    f"{class_name} must have attribute {name} so it can be used in likelihood"
                )
        if hasattr(self.functions, "jac_hf") and hasattr(self.functions, "jac_chf"):
            self.hasjac = True

    def d_contrib(self, lifetimes: Lifetimes) -> float:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return -np.sum(np.log(self.functions.hf(lifetimes.values)))

    def rc_contrib(self, lifetimes: Lifetimes) -> float:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return np.sum(self.functions.chf(lifetimes.values))

    def lc_contrib(self, lifetimes: Lifetimes) -> float:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return -np.sum(
            np.log(
                -np.expm1(
                    -self.functions.chf(
                        lifetimes.values,
                    )
                )
            )
        )

    def lt_contrib(self, lifetimes: Lifetimes) -> float:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return -np.sum(self.functions.chf(lifetimes.values))

    def jac_d_contrib(self, lifetimes: Lifetimes) -> FloatArray:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return -np.sum(
            self.functions.jac_hf(lifetimes.values)
            / self.functions.hf(lifetimes.values),
            axis=0,
        )

    def jac_rc_contrib(self, lifetimes: Lifetimes) -> FloatArray:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return np.sum(
            self.functions.jac_chf(lifetimes.values),
            axis=0,
        )

    def jac_lc_contrib(self, lifetimes: Lifetimes) -> FloatArray:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return -np.sum(
            self.functions.jac_chf(lifetimes.values)
            / np.expm1(self.functions.chf(lifetimes.values)),
            axis=0,
        )

    def jac_lt_contrib(self, lifetimes: Lifetimes) -> FloatArray:
        """

        Args:
            lifetimes ():

        Returns:

        """
        for name, data in self.kwdata.items():
            setattr(self.functions, name, data[lifetimes.index])
        return -np.sum(
            self.functions.jac_chf(lifetimes.values),
            axis=0,
        )

    def negative_log(
        self,
        params: FloatArray,
    ) -> float:
        self.params = params
        return (
            self.d_contrib(self.observed_lifetimes.complete)
            + self.rc_contrib(self.observed_lifetimes.rc)
            + self.lc_contrib(self.observed_lifetimes.left_censored)
            + self.lt_contrib(self.truncations.left)
        )

    def jac_negative_log(
        self,
        params: FloatArray,
    ) -> Union[None, FloatArray]:
        """

        Args:
            params ():

        Returns:

        """
        if not self.hasjac:
            warnings.warn("Functions does not support jac negative likelihood natively")
            return None
        self.params = params
        return (
            self.jac_d_contrib(self.observed_lifetimes.complete)
            + self.jac_rc_contrib(self.observed_lifetimes.rc)
            + self.jac_lc_contrib(self.observed_lifetimes.left_censored)
            + self.jac_lt_contrib(self.truncations.left)
        )
