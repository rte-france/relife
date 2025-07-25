import numpy as np
from pytest import approx
from relife.quadrature import legendre_quadrature, laguerre_quadrature


def test_laguerre_quadrature(integration_bound_a):
    # f : R -> R or R^n -> R^n or R^(m, n) -> R^(m, n), etc.
    f = lambda x: x
    # integral_a^inf x*exp(-x)dx = (a + 1)*exp(-a)
    expected_intg = lambda a: (a + 1) * np.exp(-a)

    integration = laguerre_quadrature(f, integration_bound_a)
    assert integration.shape == integration_bound_a.shape
    assert integration == approx(expected_intg(integration_bound_a))

    # g : R -> R^d or R^n -> R^(d, n) or R^(m, n) -> R^(d, m, n), etc.
    d = 6
    stack_f = lambda x: np.stack([x] * d, axis=0)
    integration = laguerre_quadrature(stack_f, integration_bound_a)
    assert integration.shape == (d,) + integration_bound_a.shape
    assert integration == approx(stack_f(expected_intg(integration_bound_a)))


def test_legendre_quadrature(integration_bound_a, integration_bound_b):
    m = 2
    n = 3

    # f : R -> R or R^n -> R^n or R^(m, n) -> R^(m, n), etc.
    f = lambda x: x
    # integral_a^b xdx = (1/2)*(b^2 - a^2)
    expected_intg = lambda a, b: 0.5 * (b**2 - a**2)

    integration = legendre_quadrature(f, integration_bound_a, integration_bound_b)
    assert integration.shape == np.broadcast_shapes(integration_bound_a.shape, integration_bound_b.shape)
    assert integration == approx(expected_intg(integration_bound_a, integration_bound_b))

    # f : R -> R^d or R^n -> R^(d, n) or R^(m, n) -> R^(d, m, n), etc.
    d = 6
    stack_f = lambda x: np.stack([x] * d, axis=0)
    integration = legendre_quadrature(stack_f, integration_bound_a, integration_bound_b)
    assert integration.shape == (d,) + np.broadcast_shapes(integration_bound_a.shape, integration_bound_b.shape)
    assert integration == approx(stack_f(expected_intg(integration_bound_a, integration_bound_b)))
