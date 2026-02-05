# pyright: basic

import numpy as np

from relife.stochastic_process import Kijima1Process, Kijima2Process
from relife.stochastic_process._sample._iterables import (
    Kijima1ProcessIterable, Kijima2ProcessIterable,
)
from relife.stochastic_process._sample.tests.utils import select_from_struct


class TestBasicSampling:

    def test_kijima_1(self, distribution):
        kijima_1 = Kijima1Process(distribution, q=0.5)
        nb_samples = 10
        t0 = distribution.ppf(0.3)
        tf = distribution.ppf(0.95)

        iterable = Kijima1ProcessIterable(kijima_1, nb_samples, (t0, tf))
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

        # Check Kijima I property for each sample
        for i in range(nb_samples):
            select_sample = select_from_struct(struct_array, sample_id=i)
            np.testing.assert_almost_equal(kijima_1.q * select_sample["timeline"][:-1], select_sample["entry"][1:])

    def test_kijima_2(self, distribution):
        kijima_2 = Kijima2Process(distribution, q=0.5)
        nb_samples = 10
        t0 = distribution.ppf(0.3)
        tf = distribution.ppf(0.95)

        iterable = Kijima2ProcessIterable(kijima_2, nb_samples, (t0, tf))
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

        # Check Kijima II property for each sample
        for i in range(nb_samples):
            select_sample = select_from_struct(struct_array, sample_id=i)
            np.testing.assert_almost_equal(kijima_2.q * select_sample["time"][:-1], select_sample["entry"][1:])
