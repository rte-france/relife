import numpy as np
from pytest import approx

from relife.lifetime_model import Weibull, AgeReplacementModel, LeftTruncatedModel


def test_age_replacement_model_event_logic():
    """Test that AgeReplacementModel correctly sets event=False for replacements."""
    # Create a simple Weibull model
    wei = Weibull(7, 0.05)
    
    # Freeze it with a replacement age of 20
    ar_model = AgeReplacementModel(wei).freeze(ar=20)
    
    # Generate samples
    time, event = ar_model.rvs(size=1000, return_event=True)
    
    # Check that samples at/above replacement age have event=False
    replacement_mask = time >= 20
    if np.any(replacement_mask):
        assert np.all(~event[replacement_mask]), "All samples at/above replacement age should have event=False"
    
    # Check that samples below replacement age have event=True
    below_replacement_mask = time < 20
    if np.any(below_replacement_mask):
        assert np.all(event[below_replacement_mask]), "All samples below replacement age should have event=True"


def test_age_replacement_left_truncated_combined():
    """Test combined model AgeReplacementModel(LeftTruncatedModel) event logic."""
    # Create models
    wei = Weibull(7, 0.05)
    lt_model = LeftTruncatedModel(wei).freeze(a0=10)
    combined_model = AgeReplacementModel(lt_model).freeze(ar=20)
    
    # Generate samples
    time, event, entry = combined_model.rvs(size=1000, return_event=True, return_entry=True)
    
    # Verify entry values
    assert np.all(entry == 10), "All entries should be 10"
    
    # For AgeReplacementModel(LeftTruncatedModel), the AgeReplacementModel applies to the
    # already left-truncated model, so replacement occurs when the underlying time >= ar
    replacement_mask = time >= 20
    if np.any(replacement_mask):
        # These should have event=False (replacements)
        assert np.all(~event[replacement_mask]), "Samples at/above replacement age should have event=False"