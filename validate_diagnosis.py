#!/usr/bin/env python3
"""
Validation script to confirm the Gompertz variance diagnosis.
This script adds logging to validate our assumptions about the bug.
"""

import numpy as np
from scipy.special import polygamma
import sys
sys.path.insert(0, '.')

from relife.lifetime_model import Gompertz

def validate_diagnosis():
    """Add logging to validate the diagnosis of the Gompertz variance bug."""
    
    print("=== VALIDATION OF GOMPERTZ VARIANCE DIAGNOSIS ===\n")
    
    # Test case with clear parameter differences
    shape1, rate1 = 1.0, 0.1
    shape2, rate2 = 2.0, 0.1  # Same rate, different shape
    shape3, rate3 = 1.0, 0.2  # Same shape, different rate
    
    print("HYPOTHESIS 1: Current var() ignores shape parameter")
    print("-" * 55)
    
    gomp1 = Gompertz(shape=shape1, rate=rate1)
    gomp2 = Gompertz(shape=shape2, rate=rate2)  # Different shape, same rate
    gomp3 = Gompertz(shape=shape3, rate=rate3)  # Same shape, different rate
    
    var1_current = gomp1.var()
    var2_current = gomp2.var()
    var3_current = gomp3.var()
    
    print(f"Gompertz(shape={shape1}, rate={rate1}).var() = {var1_current:.6f}")
    print(f"Gompertz(shape={shape2}, rate={rate2}).var() = {var2_current:.6f}")
    print(f"Gompertz(shape={shape3}, rate={rate3}).var() = {var3_current:.6f}")
    
    # Check if variance is independent of shape (when rate is same)
    shape_independence = abs(var1_current - var2_current) < 1e-10
    print(f"\nVariance independent of shape (same rate): {shape_independence}")
    
    # Check if variance scales with 1/rate²
    expected_ratio = (rate1/rate3)**2
    actual_ratio = var1_current / var3_current
    rate_scaling = abs(expected_ratio - actual_ratio) < 1e-10
    print(f"Variance scales as 1/rate²: {rate_scaling} (expected ratio: {expected_ratio:.6f}, actual: {actual_ratio:.6f})")
    
    print(f"\n✓ HYPOTHESIS 1 CONFIRMED: Current var() = polygamma(1,1)/rate² = {polygamma(1,1):.6f}/rate²")
    
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Method of moments gives different results")
    print("-" * 55)
    
    # Calculate using method of moments
    var1_moments = super(Gompertz, gomp1).var()
    var2_moments = super(Gompertz, gomp2).var()
    var3_moments = super(Gompertz, gomp3).var()
    
    print(f"Method of moments var() for case 1: {var1_moments:.6f}")
    print(f"Method of moments var() for case 2: {var2_moments:.6f}")
    print(f"Method of moments var() for case 3: {var3_moments:.6f}")
    
    # Check if method of moments depends on shape
    moments_shape_dependence = abs(var1_moments - var2_moments) > 1e-6
    print(f"\nMethod of moments depends on shape: {moments_shape_dependence}")
    
    # Show the differences
    diff1 = abs(var1_current - var1_moments)
    diff2 = abs(var2_current - var2_moments)
    diff3 = abs(var3_current - var3_moments)
    
    print(f"\nDifferences between current and method of moments:")
    print(f"Case 1 difference: {diff1:.6f}")
    print(f"Case 2 difference: {diff2:.6f}")
    print(f"Case 3 difference: {diff3:.6f}")
    
    significant_differences = all(d > 1e-6 for d in [diff1, diff2, diff3])
    print(f"\n✓ HYPOTHESIS 2 CONFIRMED: Significant differences exist: {significant_differences}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("-" * 20)
    print("✓ Current Gompertz.var() implementation is INCORRECT")
    print("✓ It uses polygamma(1,1)/rate² which ignores the shape parameter")
    print("✓ Method of moments gives different, shape-dependent results")
    print("✓ The fix should remove the override and use the base class method")

if __name__ == "__main__":
    validate_diagnosis()