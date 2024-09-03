"""
This module defines fundamental types of statistical models used in survival analysis

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import functools
from typing import Optional, Union, Any

import numpy as np
from numpy import ma
from scipy.optimize import newton

from .core import LifetimeModel
from .maths.integrations import gauss_legendre, quad_laguerre


def hf(obj: LifetimeModel, time: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """

    Args:
        obj ():
        time ():
        *args ():
        **kwargs ():

    Returns:

    """
    if "pdf" in obj.__class__.__dict__ and "sf" in obj.__class__.__dict__:
        return getattr(obj, "pdf")(time, *args, **kwargs) / getattr(obj, "sf")(
            time, *args, **kwargs
        )
    if "sf" in obj.__class__.__dict__:
        raise NotImplementedError(
            """
            ReLife does not implement hf as the derivate of chf yet. Consider adding it in future versions
            see: https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.misc.derivative.html
            or : https://github.com/maroba/findiff
            """
        )
    class_name = type(obj).__name__
    raise NotImplementedError(
        f"""
        {class_name} must implement hf function
        """
    )


def chf(obj: LifetimeModel, time: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """

    Args:
        obj ():
        time ():
        *args ():
        **kwargs ():

    Returns:

    """
    if "sf" in obj.__class__.__dict__:
        return -np.log(getattr(obj, "sf")(time, *args, **kwargs))
    if "pdf" in obj.__class__.__dict__ and "hf" in obj.__class__.__dict__:
        return -np.log(getattr(obj, "pdf")(time) / getattr(obj, "hf")(time))
    if "hf" in obj.__class__.__dict__:
        lower_bound = np.zeros_like(time)
        upper_bound = np.broadcast_to(
            np.asarray(getattr(obj, "isf")(np.array(1e-4), *args, **kwargs)), time.shape
        )
        masked_upper_bound: ma.MaskedArray = ma.MaskedArray(
            upper_bound, time >= obj.support_upper_bound
        )
        masked_lower_bound: ma.MaskedArray = ma.MaskedArray(
            lower_bound, time >= obj.support_upper_bound
        )

        integration = gauss_legendre(
            obj.hf,
            masked_lower_bound,
            masked_upper_bound,
            ndim=2,
        ) + quad_laguerre(
            obj.hf,
            masked_upper_bound,
            ndim=2,
        )
        return ma.filled(integration, 1.0)

    class_name = type(obj).__name__
    raise NotImplementedError(
        f"""
    {class_name} must implement chf or at least hf function
    """
    )


def sf(obj: LifetimeModel, time: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """

    Args:
        obj ():
        time ():
        *args ():
        **kwargs ():

    Returns:

    """
    if "chf" in obj.__class__.__dict__:
        return np.exp(
            -getattr(obj, "chf")(
                time,
                *args,
                **kwargs,
            )
        )
    if "pdf" in obj.__class__.__dict__ and "hf" in obj.__class__.__dict__:
        return getattr(obj, "pdf")(time, *args, **kwargs) / getattr(obj, "hf")(
            time, *args, **kwargs
        )

    class_name = type(obj).__name__
    raise NotImplementedError(
        f"""
    {class_name} must implement sf function
    """
    )


def pdf(obj: LifetimeModel, time: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """

    Args:
        obj ():
        time ():
        *args ():
        **kwargs ():

    Returns:

    """
    try:
        return getattr(obj, "sf")(time, *args, **kwargs) * getattr(obj, "hf")(
            time, *args, **kwargs
        )
    except NotImplementedError as err:
        class_name = type(obj).__name__
        raise NotImplementedError(
            f"""
        {class_name} must implement pdf or the above functions
        """
        ) from err


def mrl(obj: LifetimeModel, time: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """

    Args:
        obj ():
        time ():
        *args ():
        **kwargs ():

    Returns:

    """
    masked_time: ma.MaskedArray = ma.MaskedArray(time, time >= obj.support_upper_bound)
    upper_bound = np.broadcast_to(
        np.asarray(getattr(obj, "isf")(np.array(1e-4), *args, **kwargs)), time.shape
    )
    masked_upper_bound: ma.MaskedArray = ma.MaskedArray(
        upper_bound, time >= obj.support_upper_bound
    )

    def integrand(x):
        return (x - masked_time) * getattr(obj, "pdf")(x, *args, **kwargs)

    integration = gauss_legendre(
        integrand,
        masked_time,
        masked_upper_bound,
        ndim=2,
    ) + quad_laguerre(
        integrand,
        masked_upper_bound,
        ndim=2,
    )
    mrl_values = integration / getattr(obj, "sf")(masked_time, *args, **kwargs)
    return ma.filled(mrl_values, 0.0)


def moment(obj: LifetimeModel, n: int, *args: Any, **kwargs: Any) -> np.ndarray:
    """

    Args:
        obj ():
        n ():
        *args ():
        **kwargs ():

    Returns:

    """
    upper_bound = getattr(obj, "isf")(np.array(1e-4), *args, **kwargs)

    def integrand(x):
        return x**n * getattr(obj, "pdf")(x)

    return gauss_legendre(
        integrand, np.array(0.0), upper_bound, ndim=2
    ) + quad_laguerre(integrand, upper_bound, ndim=2)


def mean(obj: LifetimeModel, *args: Any, **kwargs: Any) -> np.ndarray:
    """

    Args:
        obj ():
        *args ():
        **kwargs ():

    Returns:

    """
    return getattr(obj, "moment")(1, *args, **kwargs)


def var(obj: LifetimeModel, *args: Any, **kwargs: Any) -> Union[float | np.ndarray]:
    """

    Args:
        obj ():
        *args ():
        **kwargs ():

    Returns:

    """
    return (
        getattr(obj, "moment")(2, *args, **kwargs)
        - getattr(obj, "moment")(1, *args, **kwargs) ** 2
    )


def isf(
    obj: LifetimeModel,
    probability: np.ndarray,
    *args: Any,
    **kwargs: Any,
):
    """

    Args:
        obj ():
        probability ():
        *args ():
        **kwargs ():

    Returns:

    """
    # ajouter isf déduit de ichf s'il existe
    return newton(
        lambda x: getattr(obj, "sf")(x, *args, **kwargs) - probability,
        x0=np.zeros_like(probability),
    )


def cdf(obj: LifetimeModel, time: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """

    Args:
        obj ():
        time ():
        *args ():
        **kwargs ():

    Returns:

    """
    return 1 - getattr(obj, "sf")(time, *args, **kwargs)


def rvs(
    obj: LifetimeModel,
    *args: Any,
    size: Optional[int] = 1,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """

    Args:
        obj ():
        *args ():
        size ():
        seed ():
        **kwargs ():

    Returns:

    """
    generator = np.random.RandomState(seed=seed)
    probability = generator.uniform(size=size)
    return getattr(obj, "isf")(probability, *args, **kwargs)


def ppf(
    obj: LifetimeModel, probability: np.ndarray, *args: Any, **kwargs: Any
) -> np.ndarray:
    """

    Args:
        obj ():
        probability ():
        *args ():
        **kwargs ():

    Returns:

    """
    return getattr(obj, "isf")(1 - probability, *args, **kwargs)


def median(obj: LifetimeModel, *args: Any, **kwargs: Any) -> np.ndarray:
    """

    Args:
        obj ():
        *args ():
        **kwargs ():

    Returns:

    """
    return getattr(obj, "ppf")(np.array(0.5), *args, **kwargs)


def default(method):

    @functools.wraps(method)
    def wrapper(self, *args: Any, **kwargs: Any):
        method_name = method.__name__
        if method_name not in globals():
            raise ValueError(f"{method_name} as no default implementation")
        return globals()[method_name](self, *args, **kwargs)

    wrapper.decorated_method = True
    return wrapper
