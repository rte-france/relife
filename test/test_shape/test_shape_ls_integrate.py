import pytest
import numpy as np
from pytest import approx


@pytest.mark.parametrize("fixture_name", ["weibull", "gompertz", "gamma", "loglogistic"])
def test_ls_integrate_distribution(distribution_map, fixture_name, a, b):
    m = 10
    n = 3
    distribution = distribution_map[fixture_name]

    # integral_a^b f(t)dt
    integration = distribution.ls_integrate(np.ones_like, a(), b())
    assert integration.shape == ()
    assert integration == approx(distribution.cdf(b()) - distribution.cdf(a()))
    # integral_0^inf t*f(t)dt
    integration = distribution.ls_integrate(lambda t: t, 0.0, np.inf)
    assert integration == approx(distribution.mean(), rel=1e-3)

    # integral_a^b f(t)dt
    integration = distribution.ls_integrate(np.ones_like, a(), b(n))
    assert integration.shape == (n,)
    assert integration == approx(distribution.cdf(b(n)) - distribution.cdf(a()))
    # integral_0^inf t*f(t)dt
    integration = distribution.ls_integrate(lambda t: t, 0.0, np.full((n,), np.inf))
    assert integration == approx(np.full((n,), distribution.mean()), rel=1e-3)

    # integral_a^b f(t)dt
    integration = distribution.ls_integrate(np.ones_like, a(n), b())
    assert integration.shape == (n,)
    assert integration == approx(distribution.cdf(b()) - distribution.cdf(a(n)))
    # integral_0^inf t*f(t)dt
    integration = distribution.ls_integrate(lambda t: t, np.full((n,), 0.0), np.inf)
    assert integration == approx(np.full((n,), distribution.mean()), rel=1e-3)

    # integral_a^b f(t)dt
    integration = distribution.ls_integrate(np.ones_like, a(n), b(n))
    assert integration.shape == (n,)
    assert integration == approx(distribution.cdf(b(n)) - distribution.cdf(a(n)))
    # integral_0^inf t*f(t)dt
    integration = distribution.ls_integrate(lambda t: t, np.full((n,), 0.0), np.full((n,), np.inf))
    assert integration == approx(np.full((n,), distribution.mean()), rel=1e-3)

    # integral_a^b f(t)dt
    integration = distribution.ls_integrate(np.ones_like, a(), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a()))
    # integral_0^inf t*f(t)dt
    integration = distribution.ls_integrate(lambda t: t, 0.0, np.full((m, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b f(t)dt
    integration = distribution.ls_integrate(np.ones_like, a(m, n), b())
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b()) - distribution.cdf(a(m, n)))
    # integral_0^inf t*f(t)dt
    integration = distribution.ls_integrate(lambda t: t, np.full((m, n), 0.0), np.inf)
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b f(t)dt
    integration = distribution.ls_integrate(np.ones_like, a(m, 1), b(1, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(1, n)) - distribution.cdf(a(m, 1)))
    # integral_0^inf t*f(t)dt
    integration = distribution.ls_integrate(lambda t: t, np.full((m, 1), 0.0), np.full((1, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b f(t)dt
    integration = distribution.ls_integrate(np.ones_like, a(1, n), b(m, 1))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, 1)) - distribution.cdf(a(1, n)))
    # integral_0^inf t*f(t)dt
    integration = distribution.ls_integrate(lambda t: t, np.full((1, n), 0.0), np.full((m, 1), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b f(t)dt
    integration = distribution.ls_integrate(np.ones_like, a(m, 1), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a(1, n)))
    # integral_0^inf t*f(t)dt
    integration = distribution.ls_integrate(lambda t: t, np.full((m, 1), 0.0), np.full((m, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b f(t)dt
    integration = distribution.ls_integrate(np.ones_like, a(1, n), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a(1, n)))
    # integral_0^inf t*f(t)dt
    integration = distribution.ls_integrate(lambda t: t, np.full((1, n), 0.0), np.full((m, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)

    # integral_a^b f(t)dt
    integration = distribution.ls_integrate(np.ones_like, a(m, n), b(m, n))
    assert integration.shape == (m, n)
    assert integration == approx(distribution.cdf(b(m, n)) - distribution.cdf(a(m, n)))
    # integral_0^inf t*f(t)dt
    integration = distribution.ls_integrate(lambda t: t, np.full((m, n), 0.0), np.full((m, n), np.inf))
    assert integration == approx(np.full((m, n), distribution.mean()), rel=1e-3)


# SEE: https://github.com/pytest-dev/pytest/issues/349
# it is currently impossible to pass fixtures as params of pytest parametrize. For regression it is easier to separate
# cases in order to avoid nasty test ids


def test_proportional_hazard_ls_integrate(proportional_hazard, a, b, covar):
    m = 10
    n = 3

    # integral_a^b f(t)dt
    integration = proportional_hazard.ls_integrate(np.ones_like, a(), b(), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(proportional_hazard.cdf(b(), covar(m)) - proportional_hazard.cdf(a(), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = proportional_hazard.ls_integrate(lambda t: t, 0.0, np.inf, covar(m))
    assert integration == approx(proportional_hazard.mean(covar(m)))

    # integral_a^b f(t)dt
    integration = proportional_hazard.ls_integrate(np.ones_like, a(), b(n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(proportional_hazard.cdf(b(n), covar(m)) - proportional_hazard.cdf(a(), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = proportional_hazard.ls_integrate(lambda t: t, 0.0, np.full((n,), np.inf), covar(m))
    assert integration == approx(np.full((m, n), proportional_hazard.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = proportional_hazard.ls_integrate(np.ones_like, a(n), b(), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(proportional_hazard.cdf(b(), covar(m)) - proportional_hazard.cdf(a(n), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = proportional_hazard.ls_integrate(lambda t: t, np.full((n,), 0.0), np.inf, covar(m))
    assert integration == approx(np.full((m, n), proportional_hazard.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = proportional_hazard.ls_integrate(np.ones_like, a(n), b(n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(proportional_hazard.cdf(b(n), covar(m)) - proportional_hazard.cdf(a(n), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = proportional_hazard.ls_integrate(lambda t: t, np.full((n,), 0.0), np.full((n,), np.inf), covar(m))
    assert integration == approx(np.full((m, n), proportional_hazard.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = proportional_hazard.ls_integrate(np.ones_like, a(m, 1), b(), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(proportional_hazard.cdf(b(), covar(m)) - proportional_hazard.cdf(a(m, 1), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = proportional_hazard.ls_integrate(lambda t: t, np.full((m, 1), 0.0), np.inf, covar(m))
    assert integration == approx(np.full((m, 1), proportional_hazard.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = proportional_hazard.ls_integrate(np.ones_like, a(), b(m, 1), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(proportional_hazard.cdf(b(m, 1), covar(m)) - proportional_hazard.cdf(a(), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = proportional_hazard.ls_integrate(lambda t: t, 0.0, np.full((m, 1), np.inf), covar(m))
    assert integration == approx(np.full((m, 1), proportional_hazard.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = proportional_hazard.ls_integrate(np.ones_like, a(m, 1), b(m, 1), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(
        proportional_hazard.cdf(b(m, 1), covar(m)) - proportional_hazard.cdf(a(m, 1), covar(m))
    )
    # integral_0^inf t*f(t)dt
    integration = proportional_hazard.ls_integrate(lambda t: t, np.full((m, 1), 0.0), np.full((m, 1), np.inf), covar(m))
    assert integration == approx(np.full((m, 1), proportional_hazard.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = proportional_hazard.ls_integrate(np.ones_like, a(), b(m, n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(proportional_hazard.cdf(b(m, n), covar(m)) - proportional_hazard.cdf(a(), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = proportional_hazard.ls_integrate(lambda t: t, 0.0, np.full((m, n), np.inf), covar(m))
    assert integration == approx(np.full((m, n), proportional_hazard.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = proportional_hazard.ls_integrate(np.ones_like, a(m, n), b(), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(proportional_hazard.cdf(b(), covar(m)) - proportional_hazard.cdf(a(m, n), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = proportional_hazard.ls_integrate(lambda t: t, np.full((m, n), 0.0), np.inf, covar(m))
    assert integration == approx(np.full((m, n), proportional_hazard.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = proportional_hazard.ls_integrate(np.ones_like, a(1, n), b(m, 1), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(
        proportional_hazard.cdf(b(m, 1), covar(m)) - proportional_hazard.cdf(a(1, n), covar(m))
    )
    # integral_0^inf t*f(t)dt
    integration = proportional_hazard.ls_integrate(lambda t: t, np.full((1, n), 0.0), np.full((m, 1), np.inf), covar(m))
    assert integration == approx(np.full((m, n), proportional_hazard.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = proportional_hazard.ls_integrate(np.ones_like, a(m, 1), b(1, n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(
        proportional_hazard.cdf(b(1, n), covar(m)) - proportional_hazard.cdf(a(m, 1), covar(m))
    )
    # integral_0^inf t*f(t)dt
    integration = proportional_hazard.ls_integrate(lambda t: t, np.full((m, 1), 0.0), np.full((1, n), np.inf), covar(m))
    assert integration == approx(np.full((m, n), proportional_hazard.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = proportional_hazard.ls_integrate(np.ones_like, a(m, n), b(m, n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(
        proportional_hazard.cdf(b(m, n), covar(m)) - proportional_hazard.cdf(a(m, n), covar(m))
    )
    # integral_0^inf t*f(t)dt
    integration = proportional_hazard.ls_integrate(lambda t: t, np.full((m, n), 0.0), np.full((m, n), np.inf), covar(m))
    assert integration == approx(np.full((m, n), proportional_hazard.mean(covar(m))))


def test_aft_ls_integrate(aft, a, b, covar):
    m = 10
    n = 3

    # integral_a^b f(t)dt
    integration = aft.ls_integrate(np.ones_like, a(), b(), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(aft.cdf(b(), covar(m)) - aft.cdf(a(), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = aft.ls_integrate(lambda t: t, 0.0, np.inf, covar(m))
    assert integration == approx(aft.mean(covar(m)))

    # integral_a^b f(t)dt
    integration = aft.ls_integrate(np.ones_like, a(), b(n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(aft.cdf(b(n), covar(m)) - aft.cdf(a(), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = aft.ls_integrate(lambda t: t, 0.0, np.full((n,), np.inf), covar(m))
    assert integration == approx(np.full((m, n), aft.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = aft.ls_integrate(np.ones_like, a(n), b(), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(aft.cdf(b(), covar(m)) - aft.cdf(a(n), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = aft.ls_integrate(lambda t: t, np.full((n,), 0.0), np.inf, covar(m))
    assert integration == approx(np.full((m, n), aft.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = aft.ls_integrate(np.ones_like, a(n), b(n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(aft.cdf(b(n), covar(m)) - aft.cdf(a(n), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = aft.ls_integrate(lambda t: t, np.full((n,), 0.0), np.full((n,), np.inf), covar(m))
    assert integration == approx(np.full((m, n), aft.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = aft.ls_integrate(np.ones_like, a(m, 1), b(), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(aft.cdf(b(), covar(m)) - aft.cdf(a(m, 1), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = aft.ls_integrate(lambda t: t, np.full((m, 1), 0.0), np.inf, covar(m))
    assert integration == approx(np.full((m, 1), aft.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = aft.ls_integrate(np.ones_like, a(), b(m, 1), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(aft.cdf(b(m, 1), covar(m)) - aft.cdf(a(), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = aft.ls_integrate(lambda t: t, 0.0, np.full((m, 1), np.inf), covar(m))
    assert integration == approx(np.full((m, 1), aft.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = aft.ls_integrate(np.ones_like, a(m, 1), b(m, 1), covar(m))
    assert integration.shape == (m, 1)
    assert integration == approx(aft.cdf(b(m, 1), covar(m)) - aft.cdf(a(m, 1), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = aft.ls_integrate(lambda t: t, np.full((m, 1), 0.0), np.full((m, 1), np.inf), covar(m))
    assert integration == approx(np.full((m, 1), aft.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = aft.ls_integrate(np.ones_like, a(), b(m, n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(aft.cdf(b(m, n), covar(m)) - aft.cdf(a(), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = aft.ls_integrate(lambda t: t, 0.0, np.full((m, n), np.inf), covar(m))
    assert integration == approx(np.full((m, n), aft.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = aft.ls_integrate(np.ones_like, a(m, n), b(), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(aft.cdf(b(), covar(m)) - aft.cdf(a(m, n), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = aft.ls_integrate(lambda t: t, np.full((m, n), 0.0), np.inf, covar(m))
    assert integration == approx(np.full((m, n), aft.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = aft.ls_integrate(np.ones_like, a(1, n), b(m, 1), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(aft.cdf(b(m, 1), covar(m)) - aft.cdf(a(1, n), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = aft.ls_integrate(lambda t: t, np.full((1, n), 0.0), np.full((m, 1), np.inf), covar(m))
    assert integration == approx(np.full((m, n), aft.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = aft.ls_integrate(np.ones_like, a(m, 1), b(1, n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(aft.cdf(b(1, n), covar(m)) - aft.cdf(a(m, 1), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = aft.ls_integrate(lambda t: t, np.full((m, 1), 0.0), np.full((1, n), np.inf), covar(m))
    assert integration == approx(np.full((m, n), aft.mean(covar(m))))

    # integral_a^b f(t)dt
    integration = aft.ls_integrate(np.ones_like, a(m, n), b(m, n), covar(m))
    assert integration.shape == (m, n)
    assert integration == approx(aft.cdf(b(m, n), covar(m)) - aft.cdf(a(m, n), covar(m)))
    # integral_0^inf t*f(t)dt
    integration = aft.ls_integrate(lambda t: t, np.full((m, n), 0.0), np.full((m, n), np.inf), covar(m))
    assert integration == approx(np.full((m, n), aft.mean(covar(m))))
