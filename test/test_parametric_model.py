import numpy as np
from pytest import approx

from relife import ParametricModel


class ModelA(ParametricModel):
    def __init__(self, x, y):
        super().__init__(x=x, y=y)


class ModelB(ParametricModel):
    def __init__(self, model: ModelA, coef: tuple[float, ...]):
        super().__init__(**{f"coef_{i+1}": v for i, v in enumerate(coef)})
        self.baseline = model


def test_model_composition():
    model_a = ModelA(1, 2)
    assert model_a.params == approx(np.array([1, 2], dtype=np.float64))
    assert model_a.params_names == ("x", "y")

    model_b = ModelB(model_a, (3, 4, 5))
    assert model_b.params == approx(np.array([3, 4, 5, 1, 2], dtype=np.float64))
    assert model_b.params_names == ("coef_1", "coef_2", "coef_3", "x", "y")

    model_a.params = np.array([2, 3])
    assert model_a.params == approx(np.array([2, 3], dtype=np.float64))

    model_b.params = np.array([2, 3, 4, 5, 6])
    assert model_b.params == approx(np.array([2, 3, 4, 5, 6], dtype=np.float64))

    assert model_b.baseline.params == approx(np.array([5, 6], dtype=np.float64))

    model_b.baseline.params = np.array([1, 2])
    assert model_b.params == approx(np.array([2, 3, 4, 1, 2], dtype=np.float64))
