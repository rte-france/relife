from pytest import approx
from scipy.stats import zscore
import numpy as np

from relife.data import LifetimeData
from relife.likelihood import maximum_likelihood_estimation

def test_distribution(distribution, power_transformer_data):
    expected_params = distribution.params.copy()
    time = power_transformer_data[0, :],
    event = power_transformer_data[1, :] == 1
    entry = power_transformer_data[2, :]
    fitted_distribution = maximum_likelihood_estimation(type(distribution)(), LifetimeData(time, event=event, entry=entry))
    assert fitted_distribution.params == approx(expected_params, rel=1e-3)

def test_regression(regression, insulator_string_data):
    time = insulator_string_data[0]
    event = insulator_string_data[1] == 1
    covar = zscore(np.column_stack([v[0] for v in insulator_string_data[3:]]))
    baseline = type(regression.baseline)()
    type(regression)(baseline).fit(
        time, covar, event=event
    )

