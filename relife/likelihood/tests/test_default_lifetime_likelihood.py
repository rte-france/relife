# pyright: basic

import numpy as np

from relife.likelihood import DefaultLifetimeLikelihood


class TestLikelihoodDistribution:
    def test_negative_log(self, distribution, power_transformer_data):
        likelihood = DefaultLifetimeLikelihood(
            distribution,
            power_transformer_data["time"],
            event=power_transformer_data["event"],
            entry=power_transformer_data["entry"],
        )
        assert isinstance(likelihood.negative_log(distribution.params), float)

    def test_jac_negative_log(self, distribution, power_transformer_data):
        likelihood = DefaultLifetimeLikelihood(
            distribution,
            power_transformer_data["time"],
            event=power_transformer_data["event"],
            entry=power_transformer_data["entry"],
        )
        assert likelihood.jac_negative_log(distribution.params).shape == (distribution.nb_params,)


class TestLikelihoodRegression:
    def test_negative_log(self, regression, insulator_string_data):
        covar = np.column_stack(
            (
                insulator_string_data["pHCl"],
                insulator_string_data["pH2SO4"],
            )
        )
        likelihood = DefaultLifetimeLikelihood(
            regression,
            insulator_string_data["time"],
            covar,
            event=insulator_string_data["event"],
            entry=insulator_string_data["entry"],
        )
        assert isinstance(likelihood.negative_log(regression.params), float)

    def test_jac_negative_log(self, regression, insulator_string_data):
        covar = np.column_stack(
            (
                insulator_string_data["pHCl"],
                insulator_string_data["pH2SO4"],
            )
        )
        likelihood = DefaultLifetimeLikelihood(
            regression,
            insulator_string_data["time"],
            covar,
            event=insulator_string_data["event"],
            entry=insulator_string_data["entry"],
        )
        assert likelihood.jac_negative_log(regression.params).shape == (regression.nb_params,)
