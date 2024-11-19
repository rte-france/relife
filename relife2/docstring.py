import inspect
from typing import Callable

from fiability.distribution import Exponential
from fiability.model import LifetimeModel
from numpydoc.docscrape import NumpyDocString


def set_lifetime_method_docstring(method: Callable) -> str:
    signature = inspect.signature(method)
    method_name = method.__name__
    print(signature)
    if hasattr(LifetimeModel, method_name):
        print(getattr(LifetimeModel, method_name))
        print(getattr(LifetimeModel, method_name).__doc__)
        doc = NumpyDocString(getattr(LifetimeModel, method_name).__doc__)
        print(doc)
        print(doc["Summary"])
        print(doc["Parameters"])
        print(doc["Attributes"])
        print(doc["Methods"])


set_lifetime_method_docstring(Exponential.hf)
