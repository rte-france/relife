#!/usr/bin/env python3
"""
Debug script to verify Gompertz variance calculation issue.
Compares gomp.var() with method of moments calculation.
"""

import numpy as np
from scipy.special import polygamma
import sys
import os

# Add the relife package to path
sys.path.insert(0, '.')

from relife.lifetime_model import Gompertz

def test_gompertz_variance():
    """Test Gompertz variance calculation against method of moments."""
    
    print("=== Debugging Gompertz Variance Calculation ===\n")
    
    # Test with different parameter values
    test_cases = [
        (1.0, 0.1),   # shape=1.0, rate=0.1
        (0.5, 0.05),  # shape=0.5, rate=0.05
        (2.0, 0.2),   # shape=2.0, rate=0.2
        (0.00865741, 0.06062632),  # From conftest.py
    ]
    
    for i, (shape, rate) in enumerate(test_cases, 1):
        print(f"Test Case {i}: shape={shape}, rate={rate}")
        print("-" * 50)
        
        # Create Gompertz distribution
        gomp = Gompertz(shape=shape, rate=rate)
        
        # Method 1: Current var() implementation
        current_var = gomp.var()
        print(f"Current var() method:     {current_var:.10f}")
        
        # Method 2: Method of moments (base class implementation)
        # This calls the base class method which uses moment(2) - moment(1)^2
        moment1 = gomp.moment(1)
        moment2 = gomp.moment(2)
        method_of_moments_var = moment2 - moment1**2
        print(f"Method of moments:        {method_of_moments_var:.10f}")
        
        # Method 3: Manual calculation using base class var() method
        base_var = super(Gompertz, gomp).var()
        print(f"Base class var() method:  {base_var:.10f}")
        
        # Show the difference
        diff_current_vs_moments = abs(current_var - method_of_moments_var)
        diff_base_vs_moments = abs(base_var - method_of_moments_var)
        
        print(f"Difference (current vs moments): {diff_current_vs_moments:.10f}")
        print(f"Difference (base vs moments):    {diff_base_vs_moments:.10f}")
        
        # Show what polygamma(1,1) gives us
        polygamma_val = polygamma(1, 1)
        current_formula_result = polygamma_val / rate**2
        print(f"polygamma(1,1):           {polygamma_val:.10f}")
        print(f"polygamma(1,1)/rate^2:    {current_formula_result:.10f}")
        
        # Check if current implementation matches the formula
        formula_matches = abs(current_var - current_formula_result) < 1e-10
        print(f"Current var matches formula: {formula_matches}")
        
        print(f"Mean (moment 1):          {moment1:.10f}")
        print(f"Second moment:            {moment2:.10f}")
        
        print("\n")

def analyze_correct_formula():
    """Analyze what the correct Gompertz variance formula should be."""
    print("=== Analysis of Correct Gompertz Variance Formula ===\n")
    
    # The Gompertz distribution has parameters (shape=c, rate=λ)
    # PDF: f(t) = c*λ*exp(λ*t)*exp(-c*(exp(λ*t)-1))
    # The variance should depend on both c and λ, not just λ
    
    print("Current implementation: var = polygamma(1, 1) / rate^2")
    print("This makes variance independent of the shape parameter!")
    print("This is clearly wrong as variance should depend on both parameters.\n")
    
    print("The correct approach should use the method of moments:")
    print("var = E[X^2] - (E[X])^2 = moment(2) - moment(1)^2")
    print("This is what the base class var() method does.\n")

if __name__ == "__main__":
    test_gompertz_variance()
    analyze_correct_formula()