import numpy as np
from pytest import approx
from relife.quadrature import legendre_quadrature, laguerre_quadrature


def test_legendre_quadrature(a, b):
    m = 10
    n = 3

    integration = legendre_quadrature(np.ones_like, a(), b())
    assert integration.shape == ()
    assert integration == approx(b() - a())

    integration = legendre_quadrature(np.ones_like, a(), b(n))
    assert integration.shape == (n,)
    assert integration == approx(b(n) - a())

    integration = legendre_quadrature(np.ones_like, a(n), b())
    assert integration.shape == (n,)
    assert integration == approx(b() - a(n))

    integration = legendre_quadrature(np.ones_like, a(n), b(n))
    assert integration.shape == (n,)
    assert integration == approx(b(n) - a(n))

    integration = legendre_quadrature(np.ones_like, a(), b(m, n), integrand_nb_assets=m)
    assert integration.shape == (m, n)
    assert integration == approx(b(m, n) - a())

    integration = legendre_quadrature(np.ones_like, a(m, n), b(), integrand_nb_assets=m)
    assert integration.shape == (m, n)
    assert integration == approx(b() - a(m, n))

    integration = legendre_quadrature(np.ones_like, a(m, 1), b(1, n), integrand_nb_assets=m)
    assert integration.shape == (m, n)
    assert integration == approx(b(1, n) - a(m, 1))

    integration = legendre_quadrature(np.ones_like, a(1, n), b(m, 1), integrand_nb_assets=m)
    assert integration.shape == (m, n)
    assert integration == approx(b(m, 1) - a(1, n))

    integration = legendre_quadrature(np.ones_like, a(m, 1), b(m, n), integrand_nb_assets=m)
    assert integration.shape == (m, n)
    assert integration == approx(b(m, n) - a(m, 1))

    integration = legendre_quadrature(np.ones_like, a(1, n), b(m, n), integrand_nb_assets=m)
    assert integration.shape == (m, n)
    assert integration == approx(b(m, n) - a(1, n))

    integration = legendre_quadrature(np.ones_like, a(m, n), b(m, n), integrand_nb_assets=m)
    assert integration.shape == (m, n)
    assert integration == approx(b(m, n) - a(m, n))


def test_laguerre_quadrature(a, b):
    m = 10
    n = 3

    integration = laguerre_quadrature(np.ones_like, a())
    assert integration.shape == ()
    assert integration == approx(np.exp(-a()))

    integration = laguerre_quadrature(np.ones_like, a(n))
    assert integration.shape == (n,)
    assert integration == approx(np.exp(-a(n)))

    integration = laguerre_quadrature(np.ones_like, a(m, n), integrand_nb_assets=m)
    assert integration.shape == (m, n)
    assert integration == approx(np.exp(-a(m, n)))
