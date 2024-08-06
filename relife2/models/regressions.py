from typing import Any, Optional, Union

from numpy.typing import ArrayLike

from relife2.data import ObservedLifetimes, Truncations, array_factory
from relife2.functions import (
    AFTFunctions,
    CovarEffect,
    LikelihoodFromLifetimes,
    ProportionalHazardFunctions,
    RegressionFunctions,
)
from relife2.typing import FloatArray

from .core import ParametricLifetimeModel, squeeze
from .distributions import Distribution


class Regression(ParametricLifetimeModel):
    """
    Facade implementation for regression models
    """

    functions: RegressionFunctions

    @property
    def coefficients(self) -> FloatArray:
        """
        Returns:
        """
        return self.covar_effect.params

    @coefficients.setter
    def coefficients(self, values: Union[list[float], tuple[float]]):
        """
        Args:
            values ():

        Returns:
        """
        if len(values) != self.functions.nb_covar:
            self.functions = type(self.functions)(
                CovarEffect(**{f"coef_{i}": v for i, v in enumerate(values)}),
                self.functions.baseline.copy(),
            )
        else:
            self.functions.covar_effect.params = values

    @squeeze
    def sf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        Args:
            time ():
            covar ():

        Returns:

        """
        time = array_factory(time)
        self.functions.covar = array_factory(covar)
        return self.functions.sf(time)

    @squeeze
    def isf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            probability ():
            covar ():

        Returns:

        """
        probability = array_factory(probability)
        self.functions.covar = array_factory(covar)
        return self.functions.isf(probability)

    @squeeze
    def hf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():

        Returns:

        """
        time = array_factory(time)
        self.functions.covar = array_factory(covar)
        return self.functions.hf(time)

    @squeeze
    def chf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():

        Returns:

        """
        time = array_factory(time)
        self.functions.covar = array_factory(covar)
        return self.functions.chf(time)

    @squeeze
    def cdf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():
        Returns:

        """
        time = array_factory(time)
        self.functions.covar = array_factory(covar)
        return self.functions.cdf(time)

    @squeeze
    def pdf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            probability ():
            covar ():

        Returns:

        """
        probability = array_factory(probability)
        self.functions.covar = array_factory(covar)
        return self.functions.pdf(probability)

    @squeeze
    def ppf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():

        Returns:

        """
        time = array_factory(time)
        self.functions.covar = array_factory(covar)
        return self.functions.ppf(time)

    @squeeze
    def mrl(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            covar ():

        Returns:

        """
        time = array_factory(time)
        self.functions.covar = array_factory(covar)
        return self.functions.mrl(time)

    @squeeze
    def ichf(
        self, cumulative_hazard_rate: ArrayLike, covar: ArrayLike
    ) -> Union[float, FloatArray]:
        """

        Args:
            covar ():
            cumulative_hazard_rate ():

        Returns:

        """
        cumulative_hazard_rate = array_factory(cumulative_hazard_rate)
        self.functions.covar = array_factory(covar)
        return self.functions.ichf(cumulative_hazard_rate)

    @squeeze
    def rvs(
        self, covar: ArrayLike, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        """

        Args:
            covar ():
            size ():
            seed ():

        Returns:

        """
        self.functions.covar = array_factory(covar)
        return self.functions.rvs(size=size, seed=seed)

    @squeeze
    def mean(self, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Returns:
            covar ():

        """
        self.functions.covar = array_factory(covar)
        return self.functions.mean()

    @squeeze
    def var(self, covar: ArrayLike) -> Union[float, FloatArray]:
        """

        Returns:
            covar ():

        """
        self.functions.covar = array_factory(covar)
        return self.functions.var()

    @squeeze
    def median(self, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        Returns:
            covar ():
        """
        self.functions.covar = array_factory(covar)
        return self.functions.median()

    def _init_likelihood(
        self,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
        **kwargs: Any,
    ) -> LikelihoodFromLifetimes:
        if "covar" not in kwargs:
            raise ValueError(
                """
                Regression likelihood expects covar as data.
                Please add covar values to kwargs.
                """
            )
        covar = kwargs["covar"]
        if covar.shape[-1] != self.functions.covar_effect.nb_params:
            optimized_functions = type(self.functions)(
                CovarEffect(**{f"coef_{i}": None for i in range(covar.shape[-1])}),
                self.functions.baseline.copy(),
            )
        else:
            optimized_functions = self.functions.copy()
        return LikelihoodFromLifetimes(
            optimized_functions,
            observed_lifetimes,
            truncations,
            covar=covar,
        )


def control_covar_args(
    coefficients: Optional[
        tuple[float | None] | list[float | None] | dict[str, float | None]
    ] = None,
) -> dict[str, float | None]:
    """

    Args:
        coefficients ():

    Returns:

    """
    if coefficients is None:
        return {"coef_0": None}
    if isinstance(coefficients, (list, tuple)):
        return {f"coef_{i}": v for i, v in enumerate(coefficients)}
    if isinstance(coefficients, dict):
        return coefficients
    raise ValueError("coefficients must be tuple, list or dict")


class ProportionalHazard(Regression):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: Distribution,
        coefficients: Optional[
            tuple[float | None] | list[float | None] | dict[str, float | None]
        ] = None,
    ):
        coefficients = control_covar_args(coefficients)
        super().__init__(
            ProportionalHazardFunctions(
                CovarEffect(**coefficients),
                baseline.functions.copy(),
            )
        )


class AFT(Regression):
    """BLABLABLABLA"""

    def __init__(
        self,
        baseline: Distribution,
        coefficients: Optional[
            tuple[float | None] | list[float | None] | dict[str, float | None]
        ] = None,
    ):
        coefficients = control_covar_args(coefficients)
        super().__init__(
            AFTFunctions(
                CovarEffect(**coefficients),
                baseline.functions.copy(),
            )
        )
