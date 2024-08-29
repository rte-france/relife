from relife2.survival.core import ParametricLifetimeModel

from relife2.survival import Distribution, Regression
from .aggregators import (
    EventSampling,
    SamplingWithAssets,
    SamplingWithoutAssets,
    aggregate_events,
)
from .iterators import DistributionIterator, RegressionIterator


def sample(functions: ParametricLifetimeModel, *args, **kwargs) -> EventSampling:
    """Factory of EventSampling sequence object"""
    if isinstance(functions, Distribution):
        values, samples_ids, assets_ids = aggregate_events(
            DistributionIterator(functions, *args, nb_assets=kwargs.get("nb_assets", 1))
        )
        if "nb_assets" in kwargs:
            sampling = SamplingWithAssets(values, samples_ids, assets_ids)
        else:
            sampling = SamplingWithoutAssets(values, samples_ids, assets_ids)
    elif isinstance(functions, Regression):
        values, samples_ids, assets_ids = aggregate_events(
            RegressionIterator(functions, *args)
        )
        sampling = SamplingWithAssets(values, samples_ids, assets_ids)
    else:
        raise ValueError
    return sampling
