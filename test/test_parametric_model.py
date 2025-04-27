import numpy as np
from numpy.ma.testutils import assert_array_equal

from relife import ParametricModel


class ModelA(ParametricModel):
    def __init__(self, x, y):
        super().__init__()
        self.new_params(x=x, y=y)


class ModelB(ParametricModel):
    def __init__(self, model : ModelA, coef : tuple[float, ...]):
        super().__init__()
        self.compose_with(baseline = model)
        self.new_params(**{f"coef_{i}" : v for i, v in enumerate(coef)})


def test_model_composition():
    model_a = ModelA(1,2)
    assert_array_equal(model_a.params, np.array([1,2], dtype=np.float64))
    assert model_a.params_names == ("x", "y")

    model_b = ModelB(model_a, (3,4,5))
    assert_array_equal(model_b.params, np.array([3,4,5,1,2], dtype=np.float64))
    assert model_b.params_names == ("coef_0", "coef_1", "coef_2", "x", "y")

    model_a.params = np.array([2,3])
    assert_array_equal(model_a.params, np.array([2,3], dtype=np.float64))

    model_b.params = np.array([2,3,4,5,6])
    assert_array_equal(model_b.params, np.array([2,3,4,5,6], dtype=np.float64))

    assert_array_equal(model_b.baseline.params, np.array([5,6], dtype=np.float64))

    model_b.baseline.params = np.array([1,2])
    assert_array_equal(model_b.params, np.array([2, 3, 4, 1, 2], dtype=np.float64))