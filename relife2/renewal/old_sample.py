from relife2.core import ParametricLifetimeModel
from relife2.fiability import Distribution, Regression
from .aggregators import (
    EventSampling,
    SamplingWithAssets,
    SamplingWithoutAssets,
    aggregate_events,
)
from .iterators import DistributionIterator, RegressionIterator


def sample(model: ParametricLifetimeModel, *args, **kwargs) -> EventSampling:
    """Factory of EventSampling sequence object"""
    if isinstance(model, Distribution):
        values, samples_ids, assets_ids = aggregate_events(
            DistributionIterator(model, *args, nb_assets=kwargs.get("nb_assets", 1))
        )
        if "nb_assets" in kwargs:
            sampling = SamplingWithAssets(values, samples_ids, assets_ids)
        else:
            sampling = SamplingWithoutAssets(values, samples_ids, assets_ids)
    elif isinstance(model, Regression):
        values, samples_ids, assets_ids = aggregate_events(
            RegressionIterator(model, *args)
        )
        sampling = SamplingWithAssets(values, samples_ids, assets_ids)
    else:
        raise ValueError
    return sampling
