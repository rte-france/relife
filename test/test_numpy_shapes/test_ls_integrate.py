import numpy as np


def test_weibull_ls_integrate(weibull, a, b):
    m = 10
    n = 3

    integration = weibull.ls_integrate(np.ones_like, a(), b())
    assert integration.shape == ()

    integration = weibull.ls_integrate(np.ones_like, a(), b((n,)))
    assert integration.shape == (n,)

    integration = weibull.ls_integrate(np.ones_like, a((n,)), b())
    assert integration.shape == (n,)

    integration = weibull.ls_integrate(np.ones_like, a((n,)), b((n,)))
    assert integration.shape == (n,)

    integration = weibull.ls_integrate(np.ones_like, a(), b((m,n)))
    assert integration.shape == (m,n)

    integration = weibull.ls_integrate(np.ones_like, a((m,n)), b())
    assert integration.shape == (m,n)

    integration = weibull.ls_integrate(np.ones_like, a((m,1)), b((1,n)))
    assert integration.shape == (m,n)

    integration = weibull.ls_integrate(np.ones_like, a((1,n)), b((m,1)))
    assert integration.shape == (m,n)

    integration = weibull.ls_integrate(np.ones_like, a((m,1)), b((m,n)))
    assert integration.shape == (m,n)

    integration = weibull.ls_integrate(np.ones_like, a((1,n)), b((m,n)))
    assert integration.shape == (m,n)

    integration = weibull.ls_integrate(np.ones_like, a((m,n)), b((m,n)))
    assert integration.shape == (m,n)


def test_gompertz_ls_integrate(gompertz, a, b):
    m = 10
    n = 3

    integration = gompertz.ls_integrate(np.ones_like, a(), b())
    assert integration.shape == ()

    integration = gompertz.ls_integrate(np.ones_like, a(), b((n,)))
    assert integration.shape == (n,)

    integration = gompertz.ls_integrate(np.ones_like, a((n,)), b())
    assert integration.shape == (n,)

    integration = gompertz.ls_integrate(np.ones_like, a((n,)), b((n,)))
    assert integration.shape == (n,)

    integration = gompertz.ls_integrate(np.ones_like, a(), b((m,n)))
    assert integration.shape == (m,n)

    integration = gompertz.ls_integrate(np.ones_like, a((m,n)), b())
    assert integration.shape == (m,n)

    integration = gompertz.ls_integrate(np.ones_like, a((m,1)), b((1,n)))
    assert integration.shape == (m,n)

    integration = gompertz.ls_integrate(np.ones_like, a((1,n)), b((m,1)))
    assert integration.shape == (m,n)

    integration = gompertz.ls_integrate(np.ones_like, a((m,1)), b((m,n)))
    assert integration.shape == (m,n)

    integration = gompertz.ls_integrate(np.ones_like, a((1,n)), b((m,n)))
    assert integration.shape == (m,n)

    integration = gompertz.ls_integrate(np.ones_like, a((m,n)), b((m,n)))
    assert integration.shape == (m,n)


def test_gamma_ls_integrate(gamma, a, b):
    m = 10
    n = 3

    integration = gamma.ls_integrate(np.ones_like, a(), b())
    assert integration.shape == ()

    integration = gamma.ls_integrate(np.ones_like, a(), b((n,)))
    assert integration.shape == (n,)

    integration = gamma.ls_integrate(np.ones_like, a((n,)), b())
    assert integration.shape == (n,)

    integration = gamma.ls_integrate(np.ones_like, a((n,)), b((n,)))
    assert integration.shape == (n,)

    integration = gamma.ls_integrate(np.ones_like, a(), b((m,n)))
    assert integration.shape == (m,n)

    integration = gamma.ls_integrate(np.ones_like, a((m,n)), b())
    assert integration.shape == (m,n)

    integration = gamma.ls_integrate(np.ones_like, a((m,1)), b((1,n)))
    assert integration.shape == (m,n)

    integration = gamma.ls_integrate(np.ones_like, a((1,n)), b((m,1)))
    assert integration.shape == (m,n)

    integration = gamma.ls_integrate(np.ones_like, a((m,1)), b((m,n)))
    assert integration.shape == (m,n)

    integration = gamma.ls_integrate(np.ones_like, a((1,n)), b((m,n)))
    assert integration.shape == (m,n)

    integration = gamma.ls_integrate(np.ones_like, a((m,n)), b((m,n)))
    assert integration.shape == (m,n)


def test_loglogistic_ls_integrate(loglogistic, a, b):
    m = 10
    n = 3

    integration = loglogistic.ls_integrate(np.ones_like, a(), b())
    assert integration.shape == ()

    integration = loglogistic.ls_integrate(np.ones_like, a(), b((n,)))
    assert integration.shape == (n,)

    integration = loglogistic.ls_integrate(np.ones_like, a((n,)), b())
    assert integration.shape == (n,)

    integration = loglogistic.ls_integrate(np.ones_like, a((n,)), b((n,)))
    assert integration.shape == (n,)

    integration = loglogistic.ls_integrate(np.ones_like, a(), b((m,n)))
    assert integration.shape == (m,n)

    integration = loglogistic.ls_integrate(np.ones_like, a((m,n)), b())
    assert integration.shape == (m,n)

    integration = loglogistic.ls_integrate(np.ones_like, a((m,1)), b((1,n)))
    assert integration.shape == (m,n)

    integration = loglogistic.ls_integrate(np.ones_like, a((1,n)), b((m,1)))
    assert integration.shape == (m,n)

    integration = loglogistic.ls_integrate(np.ones_like, a((m,1)), b((m,n)))
    assert integration.shape == (m,n)

    integration = loglogistic.ls_integrate(np.ones_like, a((1,n)), b((m,n)))
    assert integration.shape == (m,n)

    integration = loglogistic.ls_integrate(np.ones_like, a((m,n)), b((m,n)))
    assert integration.shape == (m,n)



def test_proportional_hazard_ls_integrate(proportional_hazard, a, b, covar):
    m = 10
    n = 3

    integration = proportional_hazard.ls_integrate(np.ones_like, a(), b(), covar(m))
    assert integration.shape == (m, 1)

    integration = proportional_hazard.ls_integrate(np.ones_like, a(), b((n,)), covar(m))
    assert integration.shape == (m, n)

    integration = proportional_hazard.ls_integrate(np.ones_like, a((n,)), b(), covar(m))
    assert integration.shape == (m, n)

    integration = proportional_hazard.ls_integrate(np.ones_like, a((n,)), b((n,)), covar(m))
    assert integration.shape == (m, n)

    integration = proportional_hazard.ls_integrate(np.ones_like, a((m,1)), b(), covar(m))
    assert integration.shape == (m, 1)

    integration = proportional_hazard.ls_integrate(np.ones_like, a(), b((m,1)), covar(m))
    assert integration.shape == (m, 1)

    integration = proportional_hazard.ls_integrate(np.ones_like, a((m,1)), b((m,1)), covar(m))
    assert integration.shape == (m, 1)

    integration = proportional_hazard.ls_integrate(np.ones_like, a(), b((m,n)), covar(m))
    assert integration.shape == (m, n)

    integration = proportional_hazard.ls_integrate(np.ones_like, a((m,n)), b(), covar(m))
    assert integration.shape == (m, n)

    integration = proportional_hazard.ls_integrate(np.ones_like, a((1,n)), b((m,1)), covar(m))
    assert integration.shape == (m, n)

    integration = proportional_hazard.ls_integrate(np.ones_like, a((m, 1)), b((1, n)), covar(m))
    assert integration.shape == (m, n)

    integration = proportional_hazard.ls_integrate(np.ones_like, a((m, n)), b((m, n)), covar(m))
    assert integration.shape == (m, n)


def test_aft_ls_integrate(aft, a, b, covar):
    m = 10
    n = 3

    integration = aft.ls_integrate(np.ones_like, a(), b(), covar(m))
    assert integration.shape == (m, 1)

    integration = aft.ls_integrate(np.ones_like, a(), b((n,)), covar(m))
    assert integration.shape == (m, n)

    integration = aft.ls_integrate(np.ones_like, a((n,)), b(), covar(m))
    assert integration.shape == (m, n)

    integration = aft.ls_integrate(np.ones_like, a((n,)), b((n,)), covar(m))
    assert integration.shape == (m, n)

    integration = aft.ls_integrate(np.ones_like, a((m,1)), b(), covar(m))
    assert integration.shape == (m, 1)

    integration = aft.ls_integrate(np.ones_like, a(), b((m,1)), covar(m))
    assert integration.shape == (m, 1)

    integration = aft.ls_integrate(np.ones_like, a((m,1)), b((m,1)), covar(m))
    assert integration.shape == (m, 1)

    integration = aft.ls_integrate(np.ones_like, a(), b((m,n)), covar(m))
    assert integration.shape == (m, n)

    integration = aft.ls_integrate(np.ones_like, a((m,n)), b(), covar(m))
    assert integration.shape == (m, n)

    integration = aft.ls_integrate(np.ones_like, a((1,n)), b((m,1)), covar(m))
    assert integration.shape == (m, n)

    integration = aft.ls_integrate(np.ones_like, a((m, 1)), b((1, n)), covar(m))
    assert integration.shape == (m, n)

    integration = aft.ls_integrate(np.ones_like, a((m, n)), b((m, n)), covar(m))
    assert integration.shape == (m, n)



