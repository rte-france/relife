import numpy as np
import pytest
from relife.data import LifetimeData, StructuredLifetimeData

@pytest.fixture
def lifetime_data_1d():
    time = np.array([10, 11, 9, 10, 12, 13, 11], dtype=np.float64)
    event = np.array([1, 0, 1, 0, 0, 0, 1], dtype=np.bool_)
    entry = np.array([0, 0, 3, 5, 3, 1, 9], dtype=np.float64)
    return LifetimeData(time, event=event, entry=entry)


@pytest.fixture
def lifetime_data_2d():
    time = np.array([[1, 2], [0, 4], [5, 5], [7, np.inf], [10, 10], [2, 10], [10, 11]], dtype=np.float64)
    entry = np.array([0, 0, 3, 5, 3, 1, 9], dtype=np.float64)
    departure = np.array([4, np.inf, 7, 10, np.inf, 12, np.inf], dtype=np.float64)
    return LifetimeData(time, entry=entry, departure=departure)


def test_lifetime_data(lifetime_data_1d, lifetime_data_2d):
    assert lifetime_data_1d.time.ndim == 2
    assert lifetime_data_1d.event.ndim == 2
    assert lifetime_data_1d.entry.ndim == 2
    assert lifetime_data_1d.departure.ndim == 2
    assert isinstance(lifetime_data_1d.args, tuple)
    assert lifetime_data_2d.time.ndim == 2
    assert lifetime_data_2d.event.ndim == 2
    assert lifetime_data_2d.entry.ndim == 2
    assert lifetime_data_2d.departure.ndim == 2
    assert isinstance(lifetime_data_2d.args, tuple)


def test_structured_lifetime_data_with_1d(lifetime_data_1d):
    structured_lifetime_data = StructuredLifetimeData(lifetime_data_1d)
    assert np.all(structured_lifetime_data.complete.index == np.array([0, 2, 6]).astype(np.int64))
    assert np.all(structured_lifetime_data.complete.values == np.array([10, 9, 11]).astype(np.float64).reshape(-1, 1))
    assert np.all(structured_lifetime_data.right_censoring.index == np.array([1, 3, 4, 5]))
    assert np.all(
        structured_lifetime_data.right_censoring.values == np.array([11, 10, 12, 13]).astype(np.float64).reshape(-1, 1)
    )
    assert np.all(structured_lifetime_data.left_truncation.index == np.array([2, 3, 4, 5, 6]).astype(np.int64))
    assert np.all(
        structured_lifetime_data.left_truncation.values == np.array([3, 5, 3, 1, 9]).astype(np.float64).reshape(-1, 1)
    )


def test_structured_lifetime_data_with_2d(lifetime_data_2d):
    structured_lifetime_data = StructuredLifetimeData(lifetime_data_2d)
    assert np.all(structured_lifetime_data.left_censoring.index == np.array([1]).astype(np.int64))

    assert np.all(structured_lifetime_data.left_censoring.values == np.array([4]).astype(np.float64).reshape(-1, 1))

    assert np.all(structured_lifetime_data.right_censoring.index == np.array([3]))
    assert np.all(structured_lifetime_data.right_censoring.values == np.array([7]).astype(np.float64).reshape(-1, 1))

    assert np.all(structured_lifetime_data.interval_censoring.index == np.array([0, 1, 3, 5, 6]).astype(np.int64))
    assert np.all(
        structured_lifetime_data.interval_censoring.values
        == np.array([[1, 2], [0, 4], [7, np.inf], [2, 10], [10, 11]]).astype(np.float64)
    )

    assert np.all(structured_lifetime_data.left_truncation.values == np.array([3, 5, 3, 1, 9]).reshape(-1, 1))


