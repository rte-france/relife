#!/usr/bin/env python3
"""
CRITICAL TESTS - Run these before pushing the Gompertz variance fix!
This script contains all the essential tests you MUST run to ensure the fix works.
"""

import numpy as np
import sys
import time
sys.path.insert(0, '.')

from relife.lifetime_model import Gompertz, Weibull, Gamma, Exponential

def test_1_fix_verification():
    """TEST 1: Verify the fix works correctly"""
    print("🧪 TEST 1: Fix Verification")
    print("-" * 40)
    
    gomp = Gompertz(shape=1.0, rate=0.1)
    
    # Test that var() now uses method of moments
    var_result = gomp.var()
    moment1 = gomp.moment(1)
    moment2 = gomp.moment(2)
    expected = moment2 - moment1**2
    
    diff = abs(var_result - expected)
    passed = diff < 1e-10
    
    print(f"var() result:     {var_result:.10f}")
    print(f"Method of moments: {expected:.10f}")
    print(f"Difference:       {diff:.2e}")
    print(f"✅ PASSED" if passed else f"❌ FAILED")
    return passed

def test_2_parameter_dependency():
    """TEST 2: Variance depends on both parameters"""
    print("\n🧪 TEST 2: Parameter Dependency")
    print("-" * 40)
    
    # Test shape dependency
    gomp1 = Gompertz(shape=1.0, rate=0.1)
    gomp2 = Gompertz(shape=2.0, rate=0.1)  # Different shape
    
    var1 = gomp1.var()
    var2 = gomp2.var()
    shape_dep = abs(var1 - var2) > 1e-6
    
    print(f"Shape=1.0, Rate=0.1: var = {var1:.6f}")
    print(f"Shape=2.0, Rate=0.1: var = {var2:.6f}")
    print(f"Shape dependency: {'✅ YES' if shape_dep else '❌ NO'}")
    
    # Test rate dependency  
    gomp3 = Gompertz(shape=1.0, rate=0.2)  # Different rate
    var3 = gomp3.var()
    rate_dep = abs(var1 - var3) > 1e-6
    
    print(f"Shape=1.0, Rate=0.2: var = {var3:.6f}")
    print(f"Rate dependency: {'✅ YES' if rate_dep else '❌ NO'}")
    
    return shape_dep and rate_dep

def test_3_no_crashes():
    """TEST 3: No crashes with various parameters"""
    print("\n🧪 TEST 3: No Crashes Test")
    print("-" * 40)
    
    test_params = [
        (0.001, 0.001),  # Very small
        (100, 100),      # Very large
        (0.00865741, 0.06062632),  # From conftest.py
        (0.5, 2.0),      # Mixed
    ]
    
    all_passed = True
    for shape, rate in test_params:
        try:
            gomp = Gompertz(shape=shape, rate=rate)
            var_val = gomp.var()
            mean_val = gomp.mean()
            
            # Basic sanity checks
            if var_val <= 0 or mean_val <= 0 or np.isnan(var_val) or np.isnan(mean_val):
                print(f"❌ FAILED: shape={shape}, rate={rate} - Invalid values")
                all_passed = False
            else:
                print(f"✅ PASSED: shape={shape}, rate={rate} - var={var_val:.6f}")
                
        except Exception as e:
            print(f"❌ FAILED: shape={shape}, rate={rate} - Exception: {e}")
            all_passed = False
    
    return all_passed

def test_4_other_distributions_still_work():
    """TEST 4: Other distributions not broken"""
    print("\n🧪 TEST 4: Other Distributions Still Work")
    print("-" * 40)
    
    distributions = [
        ("Weibull", Weibull(shape=2.0, rate=0.1)),
        ("Gamma", Gamma(shape=2.0, rate=0.1)),
        ("Exponential", Exponential(rate=0.1)),
    ]
    
    all_passed = True
    for name, dist in distributions:
        try:
            var_val = dist.var()
            mean_val = dist.mean()
            
            if var_val <= 0 or mean_val <= 0 or np.isnan(var_val) or np.isnan(mean_val):
                print(f"❌ FAILED: {name} - Invalid values")
                all_passed = False
            else:
                print(f"✅ PASSED: {name} - var={var_val:.6f}, mean={mean_val:.6f}")
                
        except Exception as e:
            print(f"❌ FAILED: {name} - Exception: {e}")
            all_passed = False
    
    return all_passed

def test_5_performance():
    """TEST 5: Performance is acceptable"""
    print("\n🧪 TEST 5: Performance Test")
    print("-" * 40)
    
    gomp = Gompertz(shape=1.0, rate=0.1)
    
    # Warm up
    gomp.var()
    
    # Time 1000 calls
    start_time = time.time()
    for _ in range(1000):
        gomp.var()
    elapsed = time.time() - start_time
    
    # Should be reasonable (less than 10 seconds for 1000 calls)
    acceptable = elapsed < 10.0
    
    print(f"1000 var() calls took: {elapsed:.3f} seconds")
    print(f"Average per call: {elapsed/1000*1000:.3f} ms")
    print(f"Performance: {'✅ ACCEPTABLE' if acceptable else '❌ TOO SLOW'}")
    
    return acceptable

def test_6_numerical_stability():
    """TEST 6: Numerical stability"""
    print("\n🧪 TEST 6: Numerical Stability")
    print("-" * 40)
    
    gomp = Gompertz(shape=1.0, rate=0.1)
    
    # Multiple calls should give same result
    results = [gomp.var() for _ in range(10)]
    
    # Check consistency
    max_diff = max(results) - min(results)
    stable = max_diff < 1e-12
    
    print(f"10 calls variance range: {max_diff:.2e}")
    print(f"Numerical stability: {'✅ STABLE' if stable else '❌ UNSTABLE'}")
    
    return stable

def run_all_critical_tests():
    """Run all critical tests"""
    print("🚀 RUNNING ALL CRITICAL TESTS FOR GOMPERTZ VARIANCE FIX")
    print("=" * 60)
    
    tests = [
        test_1_fix_verification,
        test_2_parameter_dependency,
        test_3_no_crashes,
        test_4_other_distributions_still_work,
        test_5_performance,
        test_6_numerical_stability,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ TEST CRASHED: {test_func.__name__} - {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("-" * 20)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"Test {i}: {status} - {test_func.__name__}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ready to push the fix!")
        print("✅ The Gompertz variance bug has been successfully fixed.")
    else:
        print(f"\n❌ {total-passed} TESTS FAILED! DO NOT push yet!")
        print("🔧 Fix the failing tests before pushing.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_critical_tests()
    sys.exit(0 if success else 1)