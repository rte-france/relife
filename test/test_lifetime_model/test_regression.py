"""
regression : LifetimeRegression

IN (time, covar)| OUT
(), (m,k) 		| (m,1)	# m assets, 1 time/asset
(n,), (m,k)	    | (m,n)	# m assets, n times/asset
(m,1), (m,k)	| (m,1)	# m assets, 1 time/asset
(m,n), (m,k) 	| (m,n)	# m assets, n times/asset
(m,n), (z,k) 	| Error # m assets in time, z assets in covar
"""

# SEE: https://github.com/pytest-dev/pytest/issues/349
# it is currently impossible to pass fixtures as params of pytest parametrize. For regression it is easier to separate
# cases in order to avoid nasty test ids

import numpy as np
from pytest import approx
from scipy.stats import boxcox, zscore

from relife.lifetime_model import Weibull, AFT, ProportionalHazard

def test_proportional_hazard(proportional_hazard, time, covar, probability):

    n = 10
    m = 3

    assert proportional_hazard.moment(1, covar(m)).shape == (m, 1)
    assert proportional_hazard.moment(2, covar(m)).shape == (m, 1)
    assert proportional_hazard.mean(covar(m)).shape == (m, 1)
    assert proportional_hazard.var(covar(m)).shape == (m, 1)
    assert proportional_hazard.median(covar(m)).shape == (m, 1)

    # covar(m).shape == (m, k)
    assert proportional_hazard.sf(time(), covar(m)).shape == (m, 1)
    assert proportional_hazard.sf(proportional_hazard.median(covar(m)), covar(m)) == approx(np.full((m,1), 0.5), rel=1e-3)
    assert proportional_hazard.hf(time(), covar(m)).shape == (m, 1)
    assert proportional_hazard.chf(time(), covar(m)).shape == (m, 1)
    assert proportional_hazard.cdf(time(), covar(m)).shape == (m, 1)
    assert proportional_hazard.pdf(time(), covar(m)).shape == (m, 1)
    assert proportional_hazard.ppf(probability(), covar(m)).shape == (m, 1)
    assert proportional_hazard.rvs(1, covar(m), seed=21).shape == (m, 1)
    assert proportional_hazard.ichf(probability(), covar(m)).shape == (m, 1)
    assert proportional_hazard.isf(probability(), covar(m)).shape == (m, 1)
    assert proportional_hazard.isf(0.5, covar(m)) == approx(proportional_hazard.median(covar(m)))

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
    assert proportional_hazard.rvs((m, 1), covar(m), seed=21).shape == (m, 1)
    assert proportional_hazard.ichf(probability(m, 1), covar(m)).shape == (m, 1)
    assert proportional_hazard.isf(probability(m, 1), covar(m)).shape == (m, 1)
    assert proportional_hazard.isf(np.full((m, 1), 0.5), covar(m)) == approx(proportional_hazard.median(covar(m)))

    # covar(m).shape == (m, k)
    assert proportional_hazard.sf(time(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.hf(time(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.chf(time(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.cdf(time(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.pdf(time(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.ppf(probability(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.rvs((m, n), covar(m), seed=21).shape == (m, n)
    assert proportional_hazard.ichf(probability(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.isf(probability(m, n), covar(m)).shape == (m, n)
    assert proportional_hazard.isf(np.full((m, n), 0.5), covar(m)) == approx(
        np.broadcast_to(proportional_hazard.median(covar(m)), (m, n))
    )


def test_aft(aft, time, covar, probability):

    n = 10
    m = 3

    assert aft.moment(1, covar(m)).shape == (m, 1)
    assert aft.moment(2, covar(m)).shape == (m, 1)
    assert aft.mean(covar(m)).shape == (m, 1)
    assert aft.var(covar(m)).shape == (m, 1)
    assert aft.median(covar(m)).shape == (m, 1)

    # covar(m).shape == (m, k)
    assert aft.sf(time(), covar(m)).shape == (m, 1)
    assert aft.sf(aft.median(covar(m)), covar(m)) == approx(np.full((m,1), 0.5), rel=1e-3)
    assert aft.hf(time(), covar(m)).shape == (m, 1)
    assert aft.chf(time(), covar(m)).shape == (m, 1)
    assert aft.cdf(time(), covar(m)).shape == (m, 1)
    assert aft.pdf(time(), covar(m)).shape == (m, 1)
    assert aft.ppf(probability(), covar(m)).shape == (m, 1)
    assert aft.rvs(1, covar(m), seed=21).shape == (m, 1)
    assert aft.ichf(probability(), covar(m)).shape == (m, 1)
    assert aft.isf(probability(), covar(m)).shape == (m, 1)
    assert aft.isf(0.5, covar(m)) == approx(aft.median(covar(m)))

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
    assert aft.rvs((m, 1), covar(m), seed=21).shape == (m, 1)
    assert aft.ichf(probability(m, 1), covar(m)).shape == (m, 1)
    assert aft.isf(probability(m, 1), covar(m)).shape == (m, 1)
    assert aft.isf(np.full((m, 1), 0.5), covar(m)) == approx(aft.median(covar(m)))

    # covar(m).shape == (m, k)
    assert aft.sf(time(m, n), covar(m)).shape == (m, n)
    assert aft.hf(time(m, n), covar(m)).shape == (m, n)
    assert aft.chf(time(m, n), covar(m)).shape == (m, n)
    assert aft.cdf(time(m, n), covar(m)).shape == (m, n)
    assert aft.pdf(time(m, n), covar(m)).shape == (m, n)
    assert aft.ppf(probability(m, n), covar(m)).shape == (m, n)
    assert aft.rvs((m, n), covar(m), seed=21).shape == (m, n)
    assert aft.ichf(probability(m, n), covar(m)).shape == (m, n)
    assert aft.isf(probability(m, n), covar(m)).shape == (m, n)
    assert aft.isf(np.full((m, n), 0.5), covar(m)) == approx(np.broadcast_to(aft.median(covar(m)), (m, n)))



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
