import numpy as np
from pytest import approx
from relife.quadrature import legendre_quadrature, laguerre_quadrature, ls_integrate



def test_laguerre_quadrature(a, b):
    m = 2
    n = 3

    # f : R -> R or R^n -> R^n or R^(m, n) -> R^(m, n), etc.
    f = lambda x : x
    # integral_a^inf x*exp(-x)dx = (a + 1)*exp(-a)
    expected_intg = lambda a: (a+1)*np.exp(-a)

    integration = laguerre_quadrature(f, a())
    assert integration.shape == ()
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


def test_ls_integrate_distribution(distribution, a, b):
    m = 2
    n = 3

    # integral_a^b dF(x)
    integration = ls_integrate(distribution, np.ones_like, a(), b())
    assert integration.shape == ()
    assert integration == approx(distribution.cdf(b()) - distribution.cdf(a()))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(distribution, lambda x: x, 0.0, np.inf, deg=100)
    assert integration == approx(distribution.mean(), rel=1e-3)

    # integral_a^b dF(x)
    integration = ls_integrate(distribution, np.ones_like, a(), b(n))
    assert integration.shape == (n,)
    assert integration == approx(distribution.cdf(b(n)) - distribution.cdf(a()))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(distribution, lambda x: x, 0.0, np.full((n,), np.inf))
    assert integration == approx(np.full((n,), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = ls_integrate(distribution, np.ones_like, a(n), b())
    assert integration.shape == (n,)
    assert integration == approx(distribution.cdf(b()) - distribution.cdf(a(n)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(distribution, lambda x: x, np.zeros(n), np.inf)
    assert integration == approx(np.full((n,), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = ls_integrate(distribution, np.ones_like, a(n), b(n))
    assert integration.shape == (n,)
    assert integration == approx(distribution.cdf(b(n)) - distribution.cdf(a(n)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(distribution, lambda x: x, np.zeros(n), np.full((n,), np.inf))
    assert integration == approx(np.full((n,), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = ls_integrate(distribution, np.ones_like, a(), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a()))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(distribution, lambda x: x, 0.0, np.full((m, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = ls_integrate(distribution, np.ones_like, a(m, n), b())
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b()) - distribution.cdf(a(m, n)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(distribution, lambda x: x, np.zeros((m, n)), np.inf)
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = ls_integrate(distribution, np.ones_like, a(m, 1), b(1, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(1, n)) - distribution.cdf(a(m, 1)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(distribution, lambda x: x, np.zeros((m,1)), np.full((1, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = ls_integrate(distribution, np.ones_like, a(1, n), b(m, 1))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, 1)) - distribution.cdf(a(1, n)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(distribution, lambda x: x, np.zeros((m,n)), np.full((m, 1), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = ls_integrate(distribution, np.ones_like, a(m, 1), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a(1, n)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(distribution, lambda x: x, np.zeros((m,1)), np.full((m, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = ls_integrate(distribution, np.ones_like, a(1, n), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a(1, n)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(distribution, lambda x: x, np.zeros((m,n)), np.full((m, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b dF(x)
    integration = ls_integrate(distribution, np.ones_like, a(m, n), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a(m, n)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(distribution, lambda x: x, np.zeros((m, n)), np.full((m, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)



def test_regression_ls_integrate(regression, a, b, covar):
    m = 2
    n = 3

    # integral_a^b dF(x)
    integration = ls_integrate(regression, np.ones_like, a(), b(), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(regression.cdf(b(), covar(m)) - regression.cdf(a(), covar(m)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(regression, lambda x: x, 0.0, np.inf, covar(m))
    assert integration == approx(regression.mean(covar(m)))

    # integral_a^b dF(x)
    integration = ls_integrate(regression, np.ones_like, a(), b(n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(regression.cdf(b(n), covar(m)) - regression.cdf(a(), covar(m)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(regression, lambda x: x, 0.0, np.full((n,), np.inf), covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = ls_integrate(regression, np.ones_like, a(n), b(), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(regression.cdf(b(), covar(m)) - regression.cdf(a(n), covar(m)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(regression, lambda x: x, np.zeros(n), np.inf, covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = ls_integrate(regression, np.ones_like, a(n), b(n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(regression.cdf(b(n), covar(m)) - regression.cdf(a(n), covar(m)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(regression, lambda x: x, np.zeros(n), np.full((n,), np.inf), covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = ls_integrate(regression, np.ones_like, a(m, 1), b(), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(regression.cdf(b(), covar(m)) - regression.cdf(a(m, 1), covar(m)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(regression, lambda x: x, np.zeros((m,1)), np.inf, covar(m))
    assert integration == approx(np.full((m, 1), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = ls_integrate(regression, np.ones_like, a(), b(m, 1), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(regression.cdf(b(m, 1), covar(m)) - regression.cdf(a(), covar(m)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(regression, lambda x: x, 0.0, np.full((m, 1), np.inf), covar(m))
    assert integration == approx(np.full((m, 1), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = ls_integrate(regression, np.ones_like, a(m, 1), b(m, 1), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(
        regression.cdf(b(m, 1), covar(m)) - regression.cdf(a(m, 1), covar(m))
    )
    # integral_0^inf x*dF(x)
    integration = ls_integrate(regression, lambda x: x, np.zeros((m,1)), np.full((m, 1), np.inf), covar(m))
    assert integration == approx(np.full((m, 1), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = ls_integrate(regression, np.ones_like, a(), b(m, n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(regression.cdf(b(m, n), covar(m)) - regression.cdf(a(), covar(m)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(regression, lambda x: x, 0.0, np.full((m, n), np.inf), covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = ls_integrate(regression, np.ones_like, a(m, n), b(), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(regression.cdf(b(), covar(m)) - regression.cdf(a(m, n), covar(m)))
    # integral_0^inf x*dF(x)
    integration = ls_integrate(regression, lambda x: x, np.zeros((m, n)), np.inf, covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = ls_integrate(regression, np.ones_like, a(1, n), b(m, 1), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(
        regression.cdf(b(m, 1), covar(m)) - regression.cdf(a(1, n), covar(m))
    )
    # integral_0^inf x*dF(x)
    integration = ls_integrate(regression, lambda x: x, np.zeros((m,n)), np.full((m, 1), np.inf), covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = ls_integrate(regression, np.ones_like, a(m, 1), b(1, n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(
        regression.cdf(b(1, n), covar(m)) - regression.cdf(a(m, 1), covar(m))
    )
    # integral_0^inf x*dF(x)
    integration = ls_integrate(regression, lambda x: x, np.zeros((m,1)), np.full((1, n), np.inf), covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))

    # integral_a^b dF(x)
    integration = ls_integrate(regression, np.ones_like, a(m, n), b(m, n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(
        regression.cdf(b(m, n), covar(m)) - regression.cdf(a(m, n), covar(m))
    )
    # integral_0^inf x*dF(x)
    integration = ls_integrate(regression, lambda x: x, np.zeros((m, n)), np.full((m, n), np.inf), covar(m))
    assert integration == approx(np.full((m, n), regression.mean(covar(m))))
