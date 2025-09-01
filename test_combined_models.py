
import numpy as np
from relife.lifetime_model import Weibull, AgeReplacementModel, LeftTruncatedModel

def test_combined_models():
    print("Testing combined models...")
    
    # Test case 1: AgeReplacementModel(LeftTruncatedModel)
    print("\n1. Testing AgeReplacementModel(LeftTruncatedModel)...")
    try:
        wei = Weibull(7, 0.05)
        lt_model = LeftTruncatedModel(wei).freeze(a0=10)
        combined_model = AgeReplacementModel(lt_model).freeze(ar=20)
        
        # This should not raise an error
        result = combined_model.rvs(size=10, return_event=True, return_entry=True)
        print("   SUCCESS: No error raised")
        print(f"   Result type: {type(result)}")
        if isinstance(result, tuple):
            time, event, entry = result
            print(f"   Time shape: {time.shape}")
            print(f"   Event shape: {event.shape}")
            print(f"   Entry shape: {entry.shape}")
            print(f"   Sample times: {time}")
            print(f"   Sample events: {event}")
            print(f"   Sample entries: {entry}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test case 2: LeftTruncatedModel(AgeReplacementModel)
    print("\n2. Testing LeftTruncatedModel(AgeReplacementModel)...")
    try:
        wei = Weibull(7, 0.05)
        ar_model = AgeReplacementModel(wei).freeze(ar=20)
        combined_model = LeftTruncatedModel(ar_model).freeze(a0=10)
        
        result = combined_model.rvs(size=10, return_event=True, return_entry=True)
        print("   SUCCESS: No error raised")
        print(f"   Result type: {type(result)}")
        if isinstance(result, tuple):
            time, event, entry = result
            print(f"   Time shape: {time.shape}")
            print(f"   Event shape: {event.shape}")
            print(f"   Entry shape: {entry.shape}")
            print(f"   Sample times: {time}")
            print(f"   Sample events: {event}")
            print(f"   Sample entries: {entry}")
            
            # Check if events are correctly set to False when time reaches ar
            # For the AgeReplacementModel, when time >= ar, event should be False
            is_replacement = time >= 20
            print(f"   Times >= 20 (replacement): {is_replacement}")
            print(f"   Events where time >= 20: {event[is_replacement] if np.any(is_replacement) else 'No replacements'}")
            print(f"   Events where time < 20: {event[~is_replacement] if np.any(~is_replacement) else 'All replacements'}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test case 3: Original issue - systematic event=True
    print("\n3. Testing for systematic event=True issue...")
    try:
        wei = Weibull(7, 0.05)
        ar_model = AgeReplacementModel(wei).freeze(ar=20)
        combined_model = LeftTruncatedModel(ar_model).freeze(a0=10)
        
        # Generate a larger sample to better see the distribution
        result = combined_model.rvs(size=100, return_event=True, return_entry=True)
        if isinstance(result, tuple):
            time, event, entry = result
            replacement_count = np.sum(time >= 20)
            false_event_count = np.sum(~event)  # Count of False events
            true_event_count = np.sum(event)    # Count of True events
            
            print(f"   Total samples: {len(time)}")
            print(f"   Samples at/above replacement age (20): {replacement_count}")
            print(f"   Samples with event=False: {false_event_count}")
            print(f"   Samples with event=True: {true_event_count}")
            
            if false_event_count > 0:
                print("   SUCCESS: Found samples with event=False (replacements)")
            else:
                print("   ISSUE: All events are True - this indicates the bug is still present")
    except Exception as e:
        print(f"   ERROR: {e}")

if __name__ == "__main__":
    test_combined_models()
