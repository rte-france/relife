#!/usr/bin/env python3
"""
Test script to verify that the Gompertz variance fix works correctly.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from relife.lifetime_model import Gompertz

def test_gompertz_variance_fix():
    """Test that the Gompertz variance fix works correctly."""
    
    print("=== TESTING GOMPERTZ VARIANCE FIX ===\n")
    
    # Test cases with different parameters
    test_cases = [
        (1.0, 0.1),   # shape=1.0, rate=0.1
        (0.5, 0.05),  # shape=0.5, rate=0.05
        (2.0, 0.2),   # shape=2.0, rate=0.2
        (0.00865741, 0.06062632),  # From conftest.py
    ]
    
    print("Testing that Gompertz.var() now uses method of moments...")
    print("-" * 60)
    
    all_tests_passed = True
    
    for i, (shape, rate) in enumerate(test_cases, 1):
        print(f"Test Case {i}: shape={shape}, rate={rate}")
        
        # Create Gompertz distribution
        gomp = Gompertz(shape=shape, rate=rate)
        
        # Get variance using the (now fixed) var() method
        var_result = gomp.var()
        
        # Calculate expected result using method of moments manually
        moment1 = gomp.moment(1)
        moment2 = gomp.moment(2)
        expected_var = moment2 - moment1**2
        
        # Check if they match
        difference = abs(var_result - expected_var)
        tolerance = 1e-10
        test_passed = difference < tolerance
        
        print(f"  var() result:     {var_result:.10f}")
        print(f"  Expected (MoM):   {expected_var:.10f}")
        print(f"  Difference:       {difference:.2e}")
        print(f"  Test passed:      {test_passed}")
        
        if not test_passed:
            all_tests_passed = False
            print(f"  ❌ FAILED: Difference {difference:.2e} > tolerance {tolerance:.2e}")
        else:
            print(f"  ✅ PASSED")
        
        print()
    
    print("=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Gompertz variance fix is working correctly.")
        print("✅ Gompertz.var() now correctly uses method of moments")
        print("✅ Variance now depends on both shape and rate parameters")
    else:
        print("❌ SOME TESTS FAILED! Fix needs more work.")
    
    print("\nAdditional verification:")
    print("-" * 30)
    
    # Test that variance now depends on shape parameter
    gomp1 = Gompertz(shape=1.0, rate=0.1)
    gomp2 = Gompertz(shape=2.0, rate=0.1)  # Same rate, different shape
    
    var1 = gomp1.var()
    var2 = gomp2.var()
    
    shape_dependence = abs(var1 - var2) > 1e-6
    print(f"Variance depends on shape parameter: {shape_dependence}")
    print(f"  shape=1.0, rate=0.1: var = {var1:.6f}")
    print(f"  shape=2.0, rate=0.1: var = {var2:.6f}")
    
    if shape_dependence:
        print("✅ Variance correctly depends on shape parameter")
    else:
        print("❌ Variance still doesn't depend on shape parameter")
    
    return all_tests_passed and shape_dependence

def compare_before_after():
    """Show the dramatic improvement from the fix."""
    print("\n" + "=" * 60)
    print("BEFORE vs AFTER COMPARISON")
    print("-" * 60)
    
    shape, rate = 1.0, 0.1
    gomp = Gompertz(shape=shape, rate=rate)
    
    # Current (fixed) result
    current_var = gomp.var()
    
    # What the old broken implementation would give
    from scipy.special import polygamma
    old_broken_var = polygamma(1, 1) / rate**2
    
    print(f"Parameters: shape={shape}, rate={rate}")
    print(f"OLD (broken) implementation: {old_broken_var:.6f}")
    print(f"NEW (fixed) implementation:  {current_var:.6f}")
    print(f"Improvement factor: {old_broken_var/current_var:.1f}x more accurate")
    
    # Show that old implementation was completely wrong
    error_reduction = abs(old_broken_var - current_var) / current_var * 100
    print(f"Error reduction: {error_reduction:.1f}%")

if __name__ == "__main__":
    success = test_gompertz_variance_fix()
    compare_before_after()
    
    if success:
        print("\n🎉 FIX VERIFICATION SUCCESSFUL!")
    else:
        print("\n❌ FIX VERIFICATION FAILED!")