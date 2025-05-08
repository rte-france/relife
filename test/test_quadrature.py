import numpy as np
from pytest import approx
from relife.quadrature import legendre_quadrature, laguerre_quadrature


def test_laguerre_quadrature(a, b):
    m = 2
    n = 3

    # f : R -> R or R^n -> R^n or R^(m, n) -> R^(m, n), etc.
    f = lambda x : x
    # integral_a^inf x*exp(-x)dx = (a + 1)*exp(-a)
    expected_intg = lambda a: (a+1)*np.exp(-a)

    integration = laguerre_quadrature(f, a())
    assert isinstance(integration, float)
    assert integration == approx(expected_intg(a()))
    integration = laguerre_quadrature(f, a(n))
    assert integration.shape == (n,)
    assert integration == approx(expected_intg(a(n)))
    integration = laguerre_quadrature(f, a(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(expected_intg(a(m,n)))

    # f : R -> R^d or R^n -> R^(d, n) or R^(m, n) -> R^(d, m, n), etc.
    d = 6
    f = lambda x: np.stack([x]*d, axis=0)
    integration = laguerre_quadrature(f, a())
    assert integration.shape == (d,)
    assert integration == approx(f(expected_intg(a())))
    integration = laguerre_quadrature(f, a(n))
    assert integration.shape == (d,n)
    assert integration == approx(f(expected_intg(a(n))))
    integration = laguerre_quadrature(f, a(m,n))
    assert integration.shape == (d,m,n)
    assert integration == approx(f(expected_intg(a(m,n))))


def test_legendre_quadrature(a, b):
    m = 2
    n = 3

    # f : R -> R or R^n -> R^n or R^(m, n) -> R^(m, n), etc.
    f = lambda x : x
    # integral_a^b xdx = (1/2)*(b^2 - a^2)
    expected_intg = lambda a, b: 0.5*(b**2 - a**2)

    integration = legendre_quadrature(f, a(), b())
    assert integration.shape == ()
    assert integration == approx(expected_intg(a(), b()))
    integration = legendre_quadrature(f, a(), b(n))
    assert integration.shape == (n,)
    assert integration == approx(expected_intg(a(), b(n)))
    integration = legendre_quadrature(f, a(n), b())
    assert integration.shape == (n,)
    assert integration == approx(expected_intg(a(n), b()))
    integration = legendre_quadrature(f, a(n), b(n))
    assert integration.shape == (n,)
    assert integration == approx(expected_intg(a(n), b(n)))
    integration = legendre_quadrature(f, a(), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(expected_intg(a(), b(m,n)))
    integration = legendre_quadrature(f, a(m, n), b())
    assert integration.shape == (m, n)
    assert integration == approx(expected_intg(a(m,n), b()))
    integration = legendre_quadrature(f, a(m, 1), b(1, n))
    assert integration.shape == (m, n)
    assert integration == approx(expected_intg(a(m,1), b(1,n)))
    integration = legendre_quadrature(f, a(1, n), b(m, 1))
    assert integration.shape == (m, n)
    assert integration == approx(expected_intg(a(1,n), b(m,1)))
    integration = legendre_quadrature(f, a(m, 1), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(0.5*(b(m, n)**2 - a(m, 1)**2))
    assert integration == approx(expected_intg(a(m,1), b(m,n)))
    assert integration.shape == (m, n)
    assert integration == approx(0.5*(b(m, n)**2 - a(1, n)**2))
    integration = legendre_quadrature(f, a(m, n), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(expected_intg(a(m,n), b(m,n)))


    # f : R -> R^d or R^n -> R^(d, n) or R^(m, n) -> R^(d, m, n), etc.
    d = 6
    f = lambda x: np.stack([x]*d, axis=0)
    # integral_a^b xdx = (1/2)*(b^2 - a^2)
    expected_intg = lambda a, b: 0.5*(b**2 - a**2)

    integration = legendre_quadrature(f, a(), b())
    assert integration.shape == (d,)
    assert integration == approx(f(expected_intg(a(), b())))
    integration = legendre_quadrature(f, a(), b(n))
    assert integration.shape == (d,n)
    assert integration == approx(f(expected_intg(a(), b(n))))
    integration = legendre_quadrature(f, a(n), b())
    assert integration.shape == (d,n)
    assert integration == approx(f(expected_intg(a(n), b())))
    integration = legendre_quadrature(f, a(n), b(n))
    assert integration.shape == (d,n)
    assert integration == approx(f(expected_intg(a(n), b(n))))
    integration = legendre_quadrature(f, a(), b(m, n))
    assert integration.shape == (d, m, n)
    assert integration == approx(f(expected_intg(a(), b(m,n))))
    integration = legendre_quadrature(f, a(m, n), b())
    assert integration.shape == (d, m, n)
    assert integration == approx(f(expected_intg(a(m,n), b())))
    integration = legendre_quadrature(f, a(m, 1), b(1, n))
    assert integration.shape == (d, m, n)
    assert integration == approx(f(expected_intg(a(m,1), b(1,n))))
    integration = legendre_quadrature(f, a(1, n), b(m, 1))
    assert integration.shape == (d, m, n)
    assert integration == approx(f(expected_intg(a(1,n), b(m,1))))
    integration = legendre_quadrature(f, a(m, 1), b(m, n))
    assert integration.shape == (d, m, n)
    assert integration == approx(f(expected_intg(a(m,1), b(m,n))))
    integration = legendre_quadrature(f, a(1, n), b(m, n))
    assert integration.shape == (d, m, n)
    assert integration == approx(f(expected_intg(a(1,n), b(m,n))))
    integration = legendre_quadrature(f, a(m, n), b(m, n))
    assert integration.shape == (d, m, n)
    assert integration == approx(f(expected_intg(a(m,n), b(m,n))))


