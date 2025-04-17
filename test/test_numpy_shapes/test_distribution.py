"""
distribution : LifetimeDistribution

IN (time)		| OUT
()			 	| ()	# 1 asset, 1 time
(n,)			| (n,)	# 1 asset, n times
(m,1)			| (m,1)	# m assets, 1 time/asset
(m, n)			| (m,n)	# m assets, n times/asset
"""
import pytest


def test_weibull(weibull, time, probability):

    assert weibull.sf(time()).shape == ()
    assert weibull.hf(time()).shape  == ()
    assert weibull.chf(time()).shape  == ()
    assert weibull.cdf(time()).shape  == ()
    assert weibull.pdf(time()).shape  == ()
    assert weibull.ppf(probability()).shape  == ()
    assert weibull.ichf(probability()).shape  == ()

    assert weibull.dhf(time()).shape  == ()
    assert weibull.jac_sf(time()).shape  == (1,2)
    assert weibull.jac_hf(time()).shape  == (1,2)
    assert weibull.jac_chf(time()).shape  == (1,2)
    assert weibull.jac_cdf(time()).shape  == (1,2)

    n = 10

    assert weibull.sf(time((n,))).shape == (n,)
    assert weibull.hf(time((n,))).shape  == (n,)
    assert weibull.chf(time((n,))).shape  == (n,)
    assert weibull.cdf(time((n,))).shape  == (n,)
    assert weibull.pdf(time((n,))).shape  == (n,)
    assert weibull.ppf(probability((n,))).shape  == (n,)
    assert weibull.ichf(probability((n,))).shape  == (n,)

    assert weibull.dhf(time(n,)).shape  == (n,)
    assert weibull.jac_sf(time(n,)).shape  == (n,2)
    assert weibull.jac_hf(time(n,)).shape  == (n,2)
    assert weibull.jac_chf(time(n,)).shape  == (n,2)
    assert weibull.jac_cdf(time(n,)).shape  == (n,2)

    m = 3

    assert weibull.sf(time((m,1))).shape == (m,1)
    assert weibull.hf(time((m,1))).shape  == (m,1)
    assert weibull.chf(time((m,1))).shape  == (m,1)
    assert weibull.cdf(time((m,1))).shape  == (m,1)
    assert weibull.pdf(time((m,1))).shape  == (m,1)
    assert weibull.ppf(probability((m,1))).shape  == (m,1)
    assert weibull.ichf(probability((m,1))).shape  == (m,1)

    assert weibull.dhf(time((m,1))).shape  == (m,1)
    assert weibull.jac_sf(time((m,1))).shape  == (m,2)
    assert weibull.jac_hf(time((m,1))).shape  == (m,2)
    assert weibull.jac_chf(time((m,1))).shape  == (m,2)
    assert weibull.jac_cdf(time((m,1))).shape  == (m,2)


    assert weibull.sf(time((m,n))).shape == (m,n)
    assert weibull.hf(time((m,n))).shape  == (m,n)
    assert weibull.chf(time((m,n))).shape  == (m,n)
    assert weibull.cdf(time((m,n))).shape  == (m,n)
    assert weibull.pdf(time((m,n))).shape  == (m,n)
    assert weibull.ppf(probability((m,n))).shape  == (m,n)
    assert weibull.ichf(probability((m,n))).shape  == (m,n)

    with pytest.raises(ValueError) as err:
        weibull.dhf(time((m, n)))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
    with pytest.raises(ValueError) as err:
        weibull.jac_sf(time((m, n)))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
    with pytest.raises(ValueError) as err:
        weibull.jac_hf(time((m, n)))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
    with pytest.raises(ValueError) as err:
        weibull.jac_chf(time((m, n)))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
    with pytest.raises(ValueError) as err:
        weibull.jac_cdf(time((m, n)))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)



def test_gompertz(gompertz, time, probability):

    assert gompertz.sf(time()).shape == ()
    assert gompertz.hf(time()).shape  == ()
    assert gompertz.chf(time()).shape  == ()
    assert gompertz.cdf(time()).shape  == ()
    assert gompertz.pdf(time()).shape  == ()
    assert gompertz.ppf(probability()).shape  == ()
    assert gompertz.ichf(probability()).shape  == ()

    n = 10

    assert gompertz.sf(time((n,))).shape == (n,)
    assert gompertz.hf(time((n,))).shape  == (n,)
    assert gompertz.chf(time((n,))).shape  == (n,)
    assert gompertz.cdf(time((n,))).shape  == (n,)
    assert gompertz.pdf(time((n,))).shape  == (n,)
    assert gompertz.ppf(probability((n,))).shape  == (n,)
    assert gompertz.ichf(probability((n,))).shape  == (n,)

    m = 3

    assert gompertz.sf(time((m,1))).shape == (m,1)
    assert gompertz.hf(time((m,1))).shape  == (m,1)
    assert gompertz.chf(time((m,1))).shape  == (m,1)
    assert gompertz.cdf(time((m,1))).shape  == (m,1)
    assert gompertz.pdf(time((m,1))).shape  == (m,1)
    assert gompertz.ppf(probability((m,1))).shape  == (m,1)
    assert gompertz.ichf(probability((m,1))).shape  == (m,1)


    assert gompertz.sf(time((m,n))).shape == (m,n)
    assert gompertz.hf(time((m,n))).shape  == (m,n)
    assert gompertz.chf(time((m,n))).shape  == (m,n)
    assert gompertz.cdf(time((m,n))).shape  == (m,n)
    assert gompertz.pdf(time((m,n))).shape  == (m,n)
    assert gompertz.ppf(probability((m,n))).shape  == (m,n)
    assert gompertz.ichf(probability((m,n))).shape  == (m,n)


def test_gamma(gamma, time, probability):

    assert gamma.sf(time()).shape == ()
    assert gamma.hf(time()).shape  == ()
    assert gamma.chf(time()).shape  == ()
    assert gamma.cdf(time()).shape  == ()
    assert gamma.pdf(time()).shape  == ()
    assert gamma.ppf(probability()).shape  == ()
    assert gamma.ichf(probability()).shape  == ()

    n = 10

    assert gamma.sf(time((n,))).shape == (n,)
    assert gamma.hf(time((n,))).shape  == (n,)
    assert gamma.chf(time((n,))).shape  == (n,)
    assert gamma.cdf(time((n,))).shape  == (n,)
    assert gamma.pdf(time((n,))).shape  == (n,)
    assert gamma.ppf(probability((n,))).shape  == (n,)
    assert gamma.ichf(probability((n,))).shape  == (n,)

    m = 3

    assert gamma.sf(time((m,1))).shape == (m,1)
    assert gamma.hf(time((m,1))).shape  == (m,1)
    assert gamma.chf(time((m,1))).shape  == (m,1)
    assert gamma.cdf(time((m,1))).shape  == (m,1)
    assert gamma.pdf(time((m,1))).shape  == (m,1)
    assert gamma.ppf(probability((m,1))).shape  == (m,1)
    assert gamma.ichf(probability((m,1))).shape  == (m,1)


    assert gamma.sf(time((m,n))).shape == (m,n)
    assert gamma.hf(time((m,n))).shape  == (m,n)
    assert gamma.chf(time((m,n))).shape  == (m,n)
    assert gamma.cdf(time((m,n))).shape  == (m,n)
    assert gamma.pdf(time((m,n))).shape  == (m,n)
    assert gamma.ppf(probability((m,n))).shape  == (m,n)
    assert gamma.ichf(probability((m,n))).shape  == (m,n)



def test_loglogistic(loglogistic, time, probability):

    assert loglogistic.sf(time()).shape == ()
    assert loglogistic.hf(time()).shape  == ()
    assert loglogistic.chf(time()).shape  == ()
    assert loglogistic.cdf(time()).shape  == ()
    assert loglogistic.pdf(time()).shape  == ()
    assert loglogistic.ppf(probability()).shape  == ()
    assert loglogistic.ichf(probability()).shape  == ()

    n = 10

    assert loglogistic.sf(time((n,))).shape == (n,)
    assert loglogistic.hf(time((n,))).shape  == (n,)
    assert loglogistic.chf(time((n,))).shape  == (n,)
    assert loglogistic.cdf(time((n,))).shape  == (n,)
    assert loglogistic.pdf(time((n,))).shape  == (n,)
    assert loglogistic.ppf(probability((n,))).shape  == (n,)
    assert loglogistic.ichf(probability((n,))).shape  == (n,)

    m = 3

    assert loglogistic.sf(time((m,1))).shape == (m,1)
    assert loglogistic.hf(time((m,1))).shape  == (m,1)
    assert loglogistic.chf(time((m,1))).shape  == (m,1)
    assert loglogistic.cdf(time((m,1))).shape  == (m,1)
    assert loglogistic.pdf(time((m,1))).shape  == (m,1)
    assert loglogistic.ppf(probability((m,1))).shape  == (m,1)
    assert loglogistic.ichf(probability((m,1))).shape  == (m,1)


    assert loglogistic.sf(time((m,n))).shape == (m,n)
    assert loglogistic.hf(time((m,n))).shape  == (m,n)
    assert loglogistic.chf(time((m,n))).shape  == (m,n)
    assert loglogistic.cdf(time((m,n))).shape  == (m,n)
    assert loglogistic.pdf(time((m,n))).shape  == (m,n)
    assert loglogistic.ppf(probability((m,n))).shape  == (m,n)
    assert loglogistic.ichf(probability((m,n))).shape  == (m,n)