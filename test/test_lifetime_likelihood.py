import numpy as np

from relife.data import LifetimeData
from relife.likelihood import LikelihoodFromLifetimes

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
