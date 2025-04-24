from relife.lifetime_model import AgeReplacementModel
import numpy as np

def test_shape_and_values_distribution(distribution, time, probability, ar):
    ar_model = AgeReplacementModel(distribution)

    assert ar_model.args_names == ("ar",)

    # frozen_model = ar_model.freeze(1)
    # frozen_model.sf(time())
    # assert ar_model.sf(time(), 1.).shape == ()


    assert ar_model.moment(1, ar(20)).shape == (20, 1)
    assert ar_model.moment(2, ar(20)).shape == (20, 1)
    assert ar_model.mean(ar(20)).shape == (20, 1)
    assert ar_model.var(ar(20)).shape == (20, 1)
    assert ar_model.median(ar(20)).shape == (20, 1)

    assert ar_model.sf(time(), ar(20)).shape == (20, 1)
    assert ar_model.hf(time(), ar(20)).shape == (20, 1)
    assert ar_model.chf(time(), ar(20)).shape == (20, 1)
    assert ar_model.cdf(time(), ar(20)).shape == (20, 1)
    assert ar_model.pdf(time(), ar(20)).shape == (20, 1)
    assert ar_model.ppf(probability(), ar(20)).shape == (20, 1)
    assert ar_model.rvs(1, ar(20), seed=21).shape == (20, 1)
    assert ar_model.ichf(probability(), ar(20)).shape == (20, 1)
    assert ar_model.isf(probability(), ar(20)).shape == (20, 1)

    n = 10
    assert ar_model.sf(time(n), ar(20)).shape == (20,n)
    assert ar_model.sf(time(n), ar(20)).shape == (20, n)
    assert ar_model.hf(time(n), ar(20)).shape ==(20, n)
    assert ar_model.chf(time(n), ar(20)).shape == (20, n)
    assert ar_model.cdf(time(n), ar(20)).shape == (20, n)
    assert ar_model.pdf(time(n), ar(20)).shape == (20, n)
    assert ar_model.ppf(probability(n), ar(20)).shape == (20, n)
    assert ar_model.rvs(n, ar(20), seed=21).shape == (20, n)
    assert ar_model.ichf(probability(n), ar(20)).shape == (20, n)
    assert ar_model.isf(probability(n), ar(20)).shape == (20, n)


    m = 10
    n = 20
    assert ar_model.sf(time(n), ar(m)).shape == (m,n)
    assert ar_model.sf(time(n), ar(m)).shape == (m,n)
    assert ar_model.hf(time(n), ar(m)).shape ==(m,n)
    assert ar_model.chf(time(n), ar(m)).shape == (m,n)
    assert ar_model.cdf(time(n), ar(m)).shape == (m,n)
    assert ar_model.pdf(time(n), ar(m)).shape == (m,n)
    assert ar_model.ppf(probability(n), ar(m)).shape == (m,n)
    assert ar_model.rvs(n, ar(m), seed=21).shape == (m,n)
    assert ar_model.ichf(probability(n), ar(m)).shape == (m,n)
    assert ar_model.isf(probability(n), ar(m)).shape == (m,n)


    assert ar_model.sf(time(m, n), ar(m)).shape == (m,n)
    assert ar_model.sf(time(m, n), ar(m)).shape == (m,n)
    assert ar_model.hf(time(m, n), ar(m)).shape ==(m,n)
    assert ar_model.chf(time(m, n), ar(m)).shape == (m,n)
    assert ar_model.cdf(time(m, n), ar(m)).shape == (m,n)
    assert ar_model.pdf(time(m, n), ar(m)).shape == (m,n)
    assert ar_model.ppf(probability(m, n), ar(m)).shape == (m,n)
    assert ar_model.rvs((m,n), ar(m), seed=21).shape == (m,n)
    assert ar_model.ichf(probability(m,n), ar(m)).shape == (m,n)
    assert ar_model.isf(probability(m,n), ar(m)).shape == (m,n)


# # TODO
# def test_shape_and_values_regression(regression, time, covar, probability, ar):
#     ar_model = AgeReplacementModel(regression)
#
#     assert ar_model.args_names == ("ar", "covar")
#
#     # frozen_model = ar_model.freeze(1)
#     # frozen_model.sf(time())
#     # assert ar_model.sf(time(), 1.).shape == ()
#
#
#     assert ar_model.moment(1, ar(20)).shape == (20, 1)
#     assert ar_model.moment(2, ar(20)).shape == (20, 1)
#     assert ar_model.mean(ar(20)).shape == (20, 1)
#     assert ar_model.var(ar(20)).shape == (20, 1)
#     assert ar_model.median(ar(20)).shape == (20, 1)
#
#     assert ar_model.sf(time(), ar(20)).shape == (20, 1)
#     assert ar_model.hf(time(), ar(20)).shape == (20, 1)
#     assert ar_model.chf(time(), ar(20)).shape == (20, 1)
#     assert ar_model.cdf(time(), ar(20)).shape == (20, 1)
#     assert ar_model.pdf(time(), ar(20)).shape == (20, 1)
#     assert ar_model.ppf(probability(), ar(20)).shape == (20, 1)
#     assert ar_model.rvs(1, ar(20), seed=21).shape == (20, 1)
#     assert ar_model.ichf(probability(), ar(20)).shape == (20, 1)
#     assert ar_model.isf(probability(), ar(20)).shape == (20, 1)
#
#     n = 10
#     assert ar_model.sf(time(n), ar(20)).shape == (20,n)
#     assert ar_model.sf(time(n), ar(20)).shape == (20, n)
#     assert ar_model.hf(time(n), ar(20)).shape ==(20, n)
#     assert ar_model.chf(time(n), ar(20)).shape == (20, n)
#     assert ar_model.cdf(time(n), ar(20)).shape == (20, n)
#     assert ar_model.pdf(time(n), ar(20)).shape == (20, n)
#     assert ar_model.ppf(probability(n), ar(20)).shape == (20, n)
#     assert ar_model.rvs(n, ar(20), seed=21).shape == (20, n)
#     assert ar_model.ichf(probability(n), ar(20)).shape == (20, n)
#     assert ar_model.isf(probability(n), ar(20)).shape == (20, n)
#
#
#     m = 10
#     n = 20
#     assert ar_model.sf(time(n), ar(m)).shape == (m,n)
#     assert ar_model.sf(time(n), ar(m)).shape == (m,n)
#     assert ar_model.hf(time(n), ar(m)).shape ==(m,n)
#     assert ar_model.chf(time(n), ar(m)).shape == (m,n)
#     assert ar_model.cdf(time(n), ar(m)).shape == (m,n)
#     assert ar_model.pdf(time(n), ar(m)).shape == (m,n)
#     assert ar_model.ppf(probability(n), ar(m)).shape == (m,n)
#     assert ar_model.rvs(n, ar(m), seed=21).shape == (m,n)
#     assert ar_model.ichf(probability(n), ar(m)).shape == (m,n)
#     assert ar_model.isf(probability(n), ar(m)).shape == (m,n)
#
#
#     assert ar_model.sf(time(m, n), ar(m)).shape == (m,n)
#     assert ar_model.sf(time(m, n), ar(m)).shape == (m,n)
#     assert ar_model.hf(time(m, n), ar(m)).shape ==(m,n)
#     assert ar_model.chf(time(m, n), ar(m)).shape == (m,n)
#     assert ar_model.cdf(time(m, n), ar(m)).shape == (m,n)
#     assert ar_model.pdf(time(m, n), ar(m)).shape == (m,n)
#     assert ar_model.ppf(probability(m, n), ar(m)).shape == (m,n)
#     assert ar_model.rvs((m,n), ar(m), seed=21).shape == (m,n)
#     assert ar_model.ichf(probability(m,n), ar(m)).shape == (m,n)
#     assert ar_model.isf(probability(m,n), ar(m)).shape == (m,n)
