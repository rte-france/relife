from relife2.functions import DistributionFunctions, RegressionFunctions
from relife2.functions.core import ParametricLifetimeFunctions

from .aggregators import (
    EventSampling,
    SamplingWithAssets,
    SamplingWithoutAssets,
    aggregate_events,
)
from .iterators import DistributionIterator, RegressionIterator


def sample(functions: ParametricLifetimeFunctions, *args, **kwargs) -> EventSampling:
    """Factory of EventSampling sequence object"""
    if isinstance(functions, DistributionFunctions):
        values, samples_ids, assets_ids = aggregate_events(
            DistributionIterator(functions, *args, nb_assets=kwargs.get("nb_assets", 1))
        )
        if "nb_assets" in kwargs:
            sampling = SamplingWithAssets(values, samples_ids, assets_ids)
        else:
            sampling = SamplingWithoutAssets(values, samples_ids, assets_ids)
    elif isinstance(functions, RegressionFunctions):
        values, samples_ids, assets_ids = aggregate_events(
            RegressionIterator(functions, *args)
        )
        sampling = SamplingWithAssets(values, samples_ids, assets_ids)
    else:
        raise ValueError
    return sampling
