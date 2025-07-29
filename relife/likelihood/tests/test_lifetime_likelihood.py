import numpy as np

from relife.data import LifetimeData
from relife.likelihood import LikelihoodFromLifetimes


class TestLikelihoodDistribution:
    def test_negative_log(self, distribution, power_transformer_data):
        lifetime_data = LifetimeData(
            power_transformer_data["time"], event=power_transformer_data["event"], entry=power_transformer_data["entry"]
        )
        likelihood = LikelihoodFromLifetimes(distribution, lifetime_data)
        assert likelihood.negative_log(distribution.params).shape == ()

    def test_jac_negative_log(self, distribution, power_transformer_data):
        lifetime_data = LifetimeData(
            power_transformer_data["time"], event=power_transformer_data["event"], entry=power_transformer_data["entry"]
        )
        likelihood = LikelihoodFromLifetimes(distribution, lifetime_data)
        assert likelihood.jac_negative_log(distribution.params).shape == (distribution.nb_params,)


class TestLikelihoodRegression:
    def test_negative_log(self, regression, insulator_string_data):
        covar = np.column_stack(
            (
                insulator_string_data["pHCl"],
                insulator_string_data["pH2SO4"],
            )
        )
        lifetime_data = LifetimeData(
            insulator_string_data["time"], insulator_string_data["event"], insulator_string_data["entry"], args=(covar,)
        )
        likelihood = LikelihoodFromLifetimes(regression, lifetime_data)
        assert likelihood.negative_log(regression.params).shape == ()

    def test_jac_negative_log(self, regression, insulator_string_data):
        covar = np.column_stack(
            (
                insulator_string_data["pHCl"],
                insulator_string_data["pH2SO4"],
            )
        )
        lifetime_data = LifetimeData(
            insulator_string_data["time"], insulator_string_data["event"], insulator_string_data["entry"], args=(covar,)
        )
        likelihood = LikelihoodFromLifetimes(regression, lifetime_data)
        assert likelihood.jac_negative_log(regression.params).shape == (regression.nb_params,)
