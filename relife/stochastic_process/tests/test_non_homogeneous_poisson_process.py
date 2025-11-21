import numpy as np

from relife.stochastic_process import NonHomogeneousPoissonProcess


class TestDistribution:

    def test_sampling(self, distribution):
        nhpp = NonHomogeneousPoissonProcess(distribution)
        n_samples = 10
        t0 = distribution.ppf(0.3)
        tf = distribution.ppf(0.95)
        sample = nhpp.sample(n_samples, (t0, tf))

        # Check NHPP proprerty for each sample
        for i in range(n_samples):
            select_sample = sample.select(sample_id=i)
            np.testing.assert_equal(select_sample.time[:-1], select_sample.entry[1:])


class TestAgeReplacementDistribution:

    def test_sampling(self, frozen_ar_distribution):
        nhpp = NonHomogeneousPoissonProcess(frozen_ar_distribution)
        ar = frozen_ar_distribution.args[0]
        n_assets = 1 if isinstance(ar, float) else ar.shape[0]
        n_samples = 10
        t0 = frozen_ar_distribution.ppf(0.3)
        tf = frozen_ar_distribution.ppf(0.95)
        sample = nhpp.sample(n_samples, (t0, tf))

        # Check that all times are less than ar for each asset
        for i in range(n_assets):
            ar_asset = ar if isinstance(ar, float) else ar[i]
            select_asset = sample.select(asset_id=i)
            np.testing.assert_array_less(select_asset.time, ar_asset + 1e-5)


class TestRegression:

    def test_sampling(self, frozen_regression):
        renewal_process = NonHomogeneousPoissonProcess(frozen_regression)
        n_assets = frozen_regression.args[0].shape[0]
        t0 = frozen_regression.ppf(0.25).min()
        tf = frozen_regression.ppf(0.95).min()
        n_samples = 10
        sample = renewal_process.sample(n_samples, (t0, tf))

        # Check NHPP proprerty for each sample and each asset
        for i in range(n_assets):
            for j in range(n_samples):
                select_sample = sample.select(asset_id=i, sample_id=j)
                np.testing.assert_equal(select_sample.time[:-1], select_sample.entry[1:])
