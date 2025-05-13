import numpy as np

from relife.data import LifetimeData
from relife.likelihood import LikelihoodFromLifetimes


def test_structured_lifetime_data_with_1d(lifetime_data_1d):

    assert np.all(lifetime_data_1d.complete.index == np.array([0, 2, 6]).astype(np.int64))
    assert np.all(lifetime_data_1d.complete.values == np.array([10, 9, 11]).astype(np.float64).reshape(-1, 1))
    assert np.all(lifetime_data_1d.right_censoring.index == np.array([1, 3, 4, 5]))
    assert np.all(
        lifetime_data_1d.right_censoring.values == np.array([11, 10, 12, 13]).astype(np.float64).reshape(-1, 1)
    )
    assert np.all(lifetime_data_1d.left_truncation.index == np.array([2, 3, 4, 5, 6]).astype(np.int64))
    assert np.all(
        lifetime_data_1d.left_truncation.values == np.array([3, 5, 3, 1, 9]).astype(np.float64).reshape(-1, 1)
    )


def test_structured_lifetime_data_with_2d(lifetime_data_2d):
    assert np.all(lifetime_data_2d.left_censoring.index == np.array([1]).astype(np.int64))

    assert np.all(lifetime_data_2d.left_censoring.values == np.array([4]).astype(np.float64).reshape(-1, 1))

    assert np.all(lifetime_data_2d.right_censoring.index == np.array([3]))
    assert np.all(lifetime_data_2d.right_censoring.values == np.array([7]).astype(np.float64).reshape(-1, 1))

    assert np.all(lifetime_data_2d.interval_censoring.index == np.array([0, 1, 3, 5, 6]).astype(np.int64))
    assert np.all(
        lifetime_data_2d.interval_censoring.values
        == np.array([[1, 2], [0, 4], [7, np.inf], [2, 10], [10, 11]]).astype(np.float64)
    )

    assert np.all(lifetime_data_2d.left_truncation.values == np.array([3, 5, 3, 1, 9]).reshape(-1, 1))


class TestLikelihoodDistribution:
    def test_negative_log(self, distribution, power_transformer_data):
        time = (power_transformer_data[0, :],)
        event = power_transformer_data[1, :] == 1
        entry = power_transformer_data[2, :]
        likelihood = LikelihoodFromLifetimes(distribution, LifetimeData(time, event=event, entry=entry))
        assert likelihood.negative_log(distribution.params).shape == ()

    def test_jac_negative_log(self, distribution, power_transformer_data):
        time = (power_transformer_data[0, :],)
        event = power_transformer_data[1, :] == 1
        entry = power_transformer_data[2, :]
        likelihood = LikelihoodFromLifetimes(distribution, LifetimeData(time, event=event, entry=entry))
        assert likelihood.jac_negative_log(distribution.params).shape == (distribution.nb_params,)


class TestLikelihoodRegression:
    def test_negative_log(self, regression, insulator_string_data):
        time = insulator_string_data[0]
        event = insulator_string_data[1] == 1
        covar = np.column_stack([v[0] for v in insulator_string_data[3:]])
        likelihood = LikelihoodFromLifetimes(regression, LifetimeData(time, event=event, args=(covar,)))
        assert likelihood.negative_log(regression.params).shape == ()

    def test_jac_negative_log(self, regression, power_transformer_data):
        time = power_transformer_data[0]
        event = power_transformer_data[1] == 1
        covar = np.column_stack([v[0] for v in power_transformer_data[3:]])
        likelihood = LikelihoodFromLifetimes(regression, LifetimeData(time, event=event, args=(covar,)))
        assert likelihood.jac_negative_log(regression.params).shape == (regression.nb_params,)
