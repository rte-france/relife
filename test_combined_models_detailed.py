import numpy as np
from relife.lifetime_model import Weibull, AgeReplacementModel, LeftTruncatedModel

def test_age_replacement_logic():
    print("Testing AgeReplacementModel event logic...")

    # Create a simple Weibull model
    wei = Weibull(7, 0.05)

    # Freeze it with a replacement age of 20
    ar_model = AgeReplacementModel(wei).freeze(ar=20)

    # Generate samples
    result = ar_model.rvs(size=1000, return_event=True)
    time, event = result

    # Check how many samples reach the replacement age
    replacement_count = np.sum(time >= 20)
    false_event_count = np.sum(~event)  # Count of False events
    true_event_count = np.sum(event)    # Count of True events

    print(f"Total samples: {len(time)}")
    print(f"Samples at/above replacement age (20): {replacement_count}")
    print(f"Samples with event=False: {false_event_count}")
    print(f"Samples with event=True: {true_event_count}")

    # For samples at/above replacement age, all should have event=False
    if replacement_count > 0:
        times_at_replacement = time[time >= 20]
        events_at_replacement = event[time >= 20]
        false_events_at_replacement = np.sum(~events_at_replacement)
        print(f"Of samples at/above replacement age, {false_events_at_replacement}/{len(times_at_replacement)} have event=False")

        if false_events_at_replacement == len(times_at_replacement):
            print("SUCCESS: All samples at/above replacement age correctly have event=False")
        else:
            print("ISSUE: Some samples at/above replacement age have event=True")

    # For samples below replacement age, all should have event=True
    below_replacement_count = np.sum(time < 20)
    if below_replacement_count > 0:
        times_below_replacement = time[time < 20]
        events_below_replacement = event[time < 20]
        true_events_below_replacement = np.sum(events_below_replacement)
        print(f"Of samples below replacement age, {true_events_below_replacement}/{len(times_below_replacement)} have event=True")

        if true_events_below_replacement == len(times_below_replacement):
            print("SUCCESS: All samples below replacement age correctly have event=True")
        else:
            print("ISSUE: Some samples below replacement age have event=False")

def test_combined_model_event_logic():
    print("\nTesting combined model event logic...")

    # Test case: LeftTruncatedModel(AgeReplacementModel)
    wei = Weibull(7, 0.05)
    ar_model = AgeReplacementModel(wei).freeze(ar=20)
    combined_model = LeftTruncatedModel(ar_model).freeze(a0=10)

    # Generate samples
    result = combined_model.rvs(size=1000, return_event=True, return_entry=True)
    time, event, entry = result

    print(f"Total samples: {len(time)}")
    print(f"Min time: {np.min(time)}, Max time: {np.max(time)}")
    print(f"All entries are 10: {np.all(entry == 10)}")

    # Check how many samples reach the replacement age (20)
    # Note: For LeftTruncatedModel, the actual age is time + a0, so replacement happens at time + 10 >= 20 => time >= 10
    replacement_count = np.sum(time >= 10)  # time + 10 >= 20
    false_event_count = np.sum(~event)  # Count of False events
    true_event_count = np.sum(event)    # Count of True events

    print(f"Samples at/above effective replacement time (10): {replacement_count}")
    print(f"Samples with event=False: {false_event_count}")
    print(f"Samples with event=True: {true_event_count}")

    # For samples at/above replacement age, all should have event=False
    if replacement_count > 0:
        times_at_replacement = time[time >= 10]
        events_at_replacement = event[time >= 10]
        false_events_at_replacement = np.sum(~events_at_replacement)
        print(f"Of samples at/above effective replacement time, {false_events_at_replacement}/{len(times_at_replacement)} have event=False")

        if false_events_at_replacement == len(times_at_replacement):
            print("SUCCESS: All samples at/above effective replacement time correctly have event=False")
        else:
            print("ISSUE: Some samples at/above effective replacement time have event=True")

if __name__ == "__main__":
    test_age_replacement_logic()
    test_combined_model_event_logic()