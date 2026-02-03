# pyright: basic

import numpy as np

from relife.stochastic_process import NonHomogeneousPoissonProcess
from relife.stochastic_process._sample._iterables import (
    NonHomogeneousPoissonProcessIterable,
)
from relife.stochastic_process._sample.tests.utils import select_from_struct


def test_basic_sampling(distribution):
    nhpp = NonHomogeneousPoissonProcess(distribution)
    nb_samples = 10
    t0 = distribution.ppf(0.3)
    tf = distribution.ppf(0.95)

    iterable = NonHomogeneousPoissonProcessIterable(nhpp, nb_samples, (t0, tf))
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # Check NHPP property for each sample
    for i in range(nb_samples):
        select_sample = select_from_struct(struct_array, sample_id=i)
        np.testing.assert_equal(select_sample["time"][:-1], select_sample["entry"][1:])


def test_age_replacement_sampling(frozen_ar_distribution):
    nhpp = NonHomogeneousPoissonProcess(frozen_ar_distribution)
    ar = frozen_ar_distribution.args[0]
    nb_assets = 1 if isinstance(ar, float) else ar.shape[0]
    nb_samples = 10
    t0 = frozen_ar_distribution.ppf(0.3)
    tf = frozen_ar_distribution.ppf(0.95)

    iterable = NonHomogeneousPoissonProcessIterable(nhpp, nb_samples, (t0, tf))
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # Check that all times are less than ar for each asset
    for i in range(nb_assets):
        ar_asset = ar if isinstance(ar, float) else ar[i]
        select_asset = select_from_struct(struct_array, asset_id=i)
        np.testing.assert_array_less(select_asset["time"], ar_asset + 1e-5)


def test_regression_sampling(frozen_regression):
    nhpp = NonHomogeneousPoissonProcess(frozen_regression)
    nb_assets = frozen_regression.args[0].shape[0]
    t0 = frozen_regression.ppf(0.25).min()
    tf = frozen_regression.ppf(0.95).min()
    nb_samples = 10

    iterable = NonHomogeneousPoissonProcessIterable(nhpp, nb_samples, (t0, tf))
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # Check NHPP proprerty for each sample and each asset
    for i in range(nb_assets):
        for j in range(nb_samples):
            select_sample = select_from_struct(struct_array, asset_id=i, sample_id=j)
            np.testing.assert_equal(
                select_sample["time"][:-1], select_sample["entry"][1:]
            )


def test_age_replacement_regression_sampling(frozen_ar_regression):
    nhpp = NonHomogeneousPoissonProcess(frozen_ar_regression)
    t0 = frozen_ar_regression.ppf(0.25).min()
    tf = 10 * frozen_ar_regression.ppf(0.75).max()
    nb_samples = 10

    iterable = NonHomogeneousPoissonProcessIterable(nhpp, nb_samples, (t0, tf))
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # check all times are bounded by the age of replacement
    # add a small constant for numerical approximations
    for i in range(frozen_ar_regression.args[0].shape[0]):
        times = select_from_struct(struct_array, asset_id=i)["time"]
        np.testing.assert_array_less(
            times, frozen_ar_regression.args[0][i].item() + 1e-5
        )
