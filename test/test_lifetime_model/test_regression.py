"""
regression : LifetimeRegression

IN (time, covar)| OUT
(), (m,k) 		| (m,1)	# m assets, 1 time/asset
(n,), (m,k)	    | (m,n)	# m assets, n times/asset
(m,1), (m,k)	| (m,1)	# m assets, 1 time/asset
(m,n), (m,k) 	| (m,n)	# m assets, n times/asset
(m,n), (z,k) 	| Error # m assets in time, z assets in covar
"""

import numpy as np
from pytest import approx
from scipy.stats import boxcox, zscore

from relife.lifetime_model import Weibull, AFT, ProportionalHazard

def test_shape_and_values(regression, time, covar, probability):

    n = 10
    m = 3

    assert regression.moment(1, covar(m)).shape == (m, 1)
    assert regression.moment(2, covar(m)).shape == (m, 1)
    assert regression.mean(covar(m)).shape == (m, 1)
    assert regression.var(covar(m)).shape == (m, 1)
    assert regression.median(covar(m)).shape == (m, 1)

    # covar(m).shape == (m, k)
    assert regression.sf(time(), covar(m)).shape == (m, 1)
    assert regression.sf(regression.median(covar(m)), covar(m)) == approx(np.full((m,1), 0.5), rel=1e-3)
    assert regression.hf(time(), covar(m)).shape == (m, 1)
    assert regression.chf(time(), covar(m)).shape == (m, 1)
    assert regression.cdf(time(), covar(m)).shape == (m, 1)
    assert regression.pdf(time(), covar(m)).shape == (m, 1)
    assert regression.ppf(probability(), covar(m)).shape == (m, 1)
    assert regression.rvs(1, covar(m), seed=21).shape == (m, 1)
    assert regression.ichf(probability(), covar(m)).shape == (m, 1)
    assert regression.isf(probability(), covar(m)).shape == (m, 1)
    assert regression.isf(0.5, covar(m)) == approx(regression.median(covar(m)))

    # covar(m).shape == (m, k)
    assert regression.sf(time(n), covar(m)).shape == (m, n)
    assert regression.hf(time(n), covar(m)).shape == (m, n)
    assert regression.chf(time(n), covar(m)).shape == (m, n)
    assert regression.cdf(time(n), covar(m)).shape == (m, n)
    assert regression.pdf(time(n), covar(m)).shape == (m, n)
    assert regression.ppf(
        probability(
            n,
        ),
        covar(m),
    ).shape == (m, n)
    assert regression.ichf(
        probability(
            n,
        ),
        covar(m),
    ).shape == (m, n)

    # covar(m).shape == (m, k)
    assert regression.sf(time(m, 1), covar(m)).shape == (m, 1)
    assert regression.hf(time(m, 1), covar(m)).shape == (m, 1)
    assert regression.chf(time(m, 1), covar(m)).shape == (m, 1)
    assert regression.cdf(time(m, 1), covar(m)).shape == (m, 1)
    assert regression.pdf(time(m, 1), covar(m)).shape == (m, 1)
    assert regression.ppf(probability(m, 1), covar(m)).shape == (m, 1)
    assert regression.rvs((m, 1), covar(m), seed=21).shape == (m, 1)
    assert regression.ichf(probability(m, 1), covar(m)).shape == (m, 1)
    assert regression.isf(probability(m, 1), covar(m)).shape == (m, 1)
    assert regression.isf(np.full((m, 1), 0.5), covar(m)) == approx(regression.median(covar(m)))

    # covar(m).shape == (m, k)
    assert regression.sf(time(m, n), covar(m)).shape == (m, n)
    assert regression.hf(time(m, n), covar(m)).shape == (m, n)
    assert regression.chf(time(m, n), covar(m)).shape == (m, n)
    assert regression.cdf(time(m, n), covar(m)).shape == (m, n)
    assert regression.pdf(time(m, n), covar(m)).shape == (m, n)
    assert regression.ppf(probability(m, n), covar(m)).shape == (m, n)
    assert regression.rvs((m, n), covar(m), seed=21).shape == (m, n)
    assert regression.ichf(probability(m, n), covar(m)).shape == (m, n)
    assert regression.isf(probability(m, n), covar(m)).shape == (m, n)
    assert regression.isf(np.full((m, n), 0.5), covar(m)) == approx(
        np.broadcast_to(regression.median(covar(m)), (m, n))
    )



def test_aft_pph_weibull_eq(insulator_string_data):
    weibull_aft = AFT(Weibull()).fit(
        insulator_string_data[0],
        zscore(np.column_stack([boxcox(v)[0] for v in insulator_string_data[3:]])),
        event=insulator_string_data[1] == 1,
    )
    weibull_pph = ProportionalHazard(Weibull()).fit(
        insulator_string_data[0],
        zscore(np.column_stack([boxcox(v)[0] for v in insulator_string_data[3:]])),
        event=insulator_string_data[1] == 1,
    )

    assert weibull_pph.baseline.params == approx(
        weibull_aft.baseline.params, rel=1e-3
    )
    assert weibull_pph.covar_effect.params == approx(
        -weibull_aft.baseline.shape * weibull_aft.covar_effect.params,
        rel=1e-3,
    )
