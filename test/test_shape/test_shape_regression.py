"""
regression : LifetimeRegression

IN (time, covar)| OUT
(), (m,k) 		| (m,1)	# m assets, 1 time/asset
(n,), (m,k)	    | (m,n)	# m assets, n times/asset
(m,1), (m,k)	| (m,1)	# m assets, 1 time/asset
(m,n), (m,k) 	| (m,n)	# m assets, n times/asset
(m,n), (z,k) 	| Error # m assets in time, z assets in covar
"""


def test_proportional_hazard(proportional_hazard, time, covar, probability):

    n = 10
    m = 3

    # covar(m).shape == (m, k)
    assert proportional_hazard.sf(time(), covar(m)).shape == (m, 1)
    assert proportional_hazard.hf(time(), covar(m)).shape == (m, 1)
    assert proportional_hazard.chf(time(), covar(m)).shape == (m, 1)
    assert proportional_hazard.cdf(time(), covar(m)).shape == (m, 1)
    assert proportional_hazard.pdf(time(), covar(m)).shape == (m, 1)
    assert proportional_hazard.ppf(probability(), covar(m)).shape == (m, 1)
    assert proportional_hazard.ichf(probability(), covar(m)).shape == (m, 1)

    # covar(m).shape == (m, k)
    assert proportional_hazard.sf(time(n), covar(m)).shape == (m, n)
    assert proportional_hazard.hf(time(n), covar(m)).shape == (m, n)
    assert proportional_hazard.chf(time(n), covar(m)).shape == (m, n)
    assert proportional_hazard.cdf(time(n), covar(m)).shape == (m, n)
    assert proportional_hazard.pdf(time(n), covar(m)).shape == (m, n)
    assert proportional_hazard.ppf(
        probability(
            n,
        ),
        covar(m),
    ).shape == (m, n)
    assert proportional_hazard.ichf(
        probability(
            n,
        ),
        covar(m),
    ).shape == (m, n)

    # covar(m).shape == (m, k)
    assert proportional_hazard.sf(time(m, 1), covar(m)).shape == (m, 1)
    assert proportional_hazard.hf(time(m, 1), covar(m)).shape == (m, 1)
    assert proportional_hazard.chf(time(m, 1), covar(m)).shape == (m, 1)
    assert proportional_hazard.cdf(time(m, 1), covar(m)).shape == (m, 1)
    assert proportional_hazard.pdf(time(m, 1), covar(m)).shape == (m, 1)
    assert proportional_hazard.ppf(probability(m, 1), covar(m)).shape == (m, 1)
    assert proportional_hazard.ichf(probability(m, 1), covar(m)).shape == (m, 1)

    # covar(m).shape == (m, k)
    assert proportional_hazard.sf(time(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.hf(time(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.chf(time(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.cdf(time(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.pdf(time(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.ppf(probability(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.ichf(probability(m, n), covar(m)).shape == (m, n)


def test_aft(aft, time, covar, probability):

    n = 10
    m = 3

    # covar(m).shape == (m, k)
    assert aft.sf(time(), covar(m)).shape == (m, 1)
    assert aft.hf(time(), covar(m)).shape == (m, 1)
    assert aft.chf(time(), covar(m)).shape == (m, 1)
    assert aft.cdf(time(), covar(m)).shape == (m, 1)
    assert aft.pdf(time(), covar(m)).shape == (m, 1)
    assert aft.ppf(probability(), covar(m)).shape == (m, 1)
    assert aft.ichf(probability(), covar(m)).shape == (m, 1)

    # covar(m).shape == (m, k)
    assert aft.sf(time(n), covar(m)).shape == (m, n)
    assert aft.hf(time(n), covar(m)).shape == (m, n)
    assert aft.chf(time(n), covar(m)).shape == (m, n)
    assert aft.cdf(time(n), covar(m)).shape == (m, n)
    assert aft.pdf(time(n), covar(m)).shape == (m, n)
    assert aft.ppf(
        probability(
            n,
        ),
        covar(m),
    ).shape == (m, n)
    assert aft.ichf(
        probability(
            n,
        ),
        covar(m),
    ).shape == (m, n)

    # covar(m).shape == (m, k)
    assert aft.sf(time(m, 1), covar(m)).shape == (m, 1)
    assert aft.hf(time(m, 1), covar(m)).shape == (m, 1)
    assert aft.chf(time(m, 1), covar(m)).shape == (m, 1)
    assert aft.cdf(time(m, 1), covar(m)).shape == (m, 1)
    assert aft.pdf(time(m, 1), covar(m)).shape == (m, 1)
    assert aft.ppf(probability(m, 1), covar(m)).shape == (m, 1)
    assert aft.ichf(probability(m, 1), covar(m)).shape == (m, 1)

    # covar(m).shape == (m, k)
    assert aft.sf(time(m, n), covar(m)).shape == (m, n)
    assert aft.hf(time(m, n), covar(m)).shape == (m, n)
    assert aft.chf(time(m, n), covar(m)).shape == (m, n)
    assert aft.cdf(time(m, n), covar(m)).shape == (m, n)
    assert aft.pdf(time(m, n), covar(m)).shape == (m, n)
    assert aft.ppf(probability(m, n), covar(m)).shape == (m, n)
    assert aft.ichf(probability(m, n), covar(m)).shape == (m, n)
