from relife.data import LifetimeData


def test_lifetime_data(lifetime_input):
    lifetime_data = LifetimeData(**lifetime_input)
    assert lifetime_data.time.ndim == 2
    assert lifetime_data.event.ndim == 2
    assert lifetime_data.entry.ndim == 2
    assert lifetime_data.departure.ndim == 2
    assert isinstance(lifetime_data.args, tuple)
