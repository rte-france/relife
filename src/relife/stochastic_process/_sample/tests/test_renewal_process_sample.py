# pyright: basic

import numpy as np

from relife.lifetime_model import EquilibriumDistribution, LeftTruncatedModel
from relife.lifetime_model._conditional_model import AgeReplacementModel
from relife.stochastic_process import RenewalProcess
from relife.stochastic_process._sample._iterables import RenewalProcessIterable
from relife.stochastic_process._sample.tests.utils import select_from_struct
from relife.utils._model_checks import get_model_nb_assets


def test_age_replacement_sampling(distribution, ar):
    renewal_process = RenewalProcess(
        distribution,
        first_lifetime_model=EquilibriumDistribution(distribution),
    )

    trial_model = AgeReplacementModel(distribution).freeze(ar)
    nb_assets = get_model_nb_assets(trial_model)
    ar_reshaped = trial_model.args[0]

    nb_samples = 10
    t0 = distribution.ppf(0.25)
    tf = 10 * distribution.ppf(0.75)

    iterable = RenewalProcessIterable(renewal_process, nb_samples, (t0, tf), ar=ar)
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # Check that all times are less than ar for each asset
    for i in range(nb_assets):
        ar_asset = np.atleast_1d(ar_reshaped)[i]
        select_asset = select_from_struct(struct_array, asset_id=i)
        assert (select_asset["time"] <= ar_asset + 1e-5).all()


def test_left_truncated_sampling(distribution, a0):
    first_lifetime_model = LeftTruncatedModel(distribution).freeze(a0)
    renewal_process = RenewalProcess(
        distribution, first_lifetime_model=first_lifetime_model
    )
    tf = 10 * distribution.ppf(0.75)
    nb_samples = 100

    iterable = RenewalProcessIterable(renewal_process, nb_samples, (0, tf))
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # check first entries are a0 for each sample
    for i in range(nb_samples):
        select_sample = select_from_struct(struct_array, sample_id=i)
        first_entries = select_sample["entry"][select_sample["entry"] > 0].reshape(
            a0.shape
        )
        np.testing.assert_equal(first_entries, a0)


def test_age_replacement_regression_sampling(frozen_regression, ar):
    renewal_process = RenewalProcess(
        frozen_regression,
        first_lifetime_model=EquilibriumDistribution(frozen_regression),
    )

    trial_model = AgeReplacementModel(frozen_regression).freeze(ar)
    nb_assets = get_model_nb_assets(trial_model)
    ar_reshaped = trial_model.args[0]

    t0 = frozen_regression.ppf(0.25).min()
    tf = 10 * frozen_regression.ppf(0.75).max()
    nb_samples = 10

    iterable = RenewalProcessIterable(renewal_process, nb_samples, (t0, tf), ar=ar)
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # check all times are bounded by the age of replacement
    # add a small constant for numerical approximations
    for i in range(nb_assets):
        ar_asset = np.atleast_1d(ar_reshaped)[i]
        times = select_from_struct(struct_array, asset_id=i)["time"]
        assert (times <= ar_asset + 1e-5).all()
