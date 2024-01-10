import pytest

from relife.datasets import load_insulator_chain_deterioration
from relife.gamma_process import  GammaProcess


@pytest.fixture(scope="module")
def data():
    return load_insulator_chain_deterioration()

@pytest.fixture(scope="module")
def process():
    return GammaProcess(10, 1)

def test_fit(process, data):

    ids, inspection_times, deterioration_measurements, increments = data
    process.fit(inspection_times=inspection_times,
                deterioration_measurements=deterioration_measurements,
                ids=ids,
                increments=increments,
                method='likelihood')

    assert (process.shape_rate.round(2), process.shape_power.round(2), process.rate.round(2)) == (3.24, 1.01, 8.74)