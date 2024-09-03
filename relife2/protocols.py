from typing import Protocol, Any, Optional, Union, runtime_checkable

import numpy as np


@runtime_checkable
class LifetimeProtocol(Protocol):
    def hf(self, time: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray: ...

    def chf(self, time: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray: ...

    def sf(self, time: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray: ...

    def pdf(self, time: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray: ...

    def mrl(self, time: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray: ...

    def mean(self, *args: Any, **kwargs: Any) -> np.ndarray: ...

    def var(self, *args: Any, **kwargs: Any) -> Union[float | np.ndarray]: ...

    def isf(
        self,
        probability: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ): ...

    def cdf(self, time: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray: ...

    def rvs(
        self,
        *args: Any,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray: ...

    def ppf(self, probability: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray: ...

    def median(self, *args: Any, **kwargs: Any) -> np.ndarray: ...


@runtime_checkable
class ParametricModelProtocol(Protocol):

    def fit(self, *args, **kwargs): ...
