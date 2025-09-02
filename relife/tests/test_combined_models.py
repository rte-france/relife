import numpy as np
import pytest
from relife.lifetime_model.conditional_model import AgeReplacementModel, LeftTruncatedModel
from relife.lifetime_model.distribution import Weibull


def test_combined_models():
    # Test AgeReplacementModel + LeftTruncatedModel
    wei = Weibull(7, 0.05)
    ar_model = AgeReplacementModel(wei).freeze(ar=20)
    combined_model = LeftTruncatedModel(ar_model).freeze(a0=10)
    time, event, entry = combined_model.rvs(size=10, return_event=True, return_entry=True)
    assert np.all((time <= 20) & (time >= 10))
    assert np.all((event == False) == (time == 20))
    assert np.all(entry == 10)

    # Test LeftTruncatedModel + AgeReplacementModel
    lt_model = LeftTruncatedModel(wei).freeze(a0=10)
    combined_model2 = AgeReplacementModel(lt_model).freeze(ar=20)
    time2, event2, entry2 = combined_model2.rvs(size=10, return_event=True, return_entry=True)
    assert np.all((time2 <= 20) & (time2 >= 10))
    assert np.all((event2 == False) == (time2 == 20))
    assert np.all(entry2 == 10)  # entry for LeftTruncatedModel is always a0
