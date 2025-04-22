import pytest
from pytest import approx
import numpy as np

@pytest.mark.parametrize("fixture_name", ["exponential", "weibull", "gompertz", "gamma", "loglogistic"])
def test_frozen_distribution(distribution_map, fixture_name, time, probability):

    distribution = distribution_map[fixture_name]
    frozen_distribution = distribution.freeze()
    assert frozen_distribution.nb_assets == 1
    assert frozen_distribution.args == ()

    assert frozen_distribution.moment(1).shape == ()
    assert frozen_distribution.moment(2).shape == ()
    assert frozen_distribution.mean().shape == ()
    assert frozen_distribution.var().shape == ()
    assert frozen_distribution.median().shape == ()

    assert frozen_distribution.sf(time()).shape == ()
    assert frozen_distribution.hf(time()).shape == ()
    assert frozen_distribution.chf(time()).shape == ()
    assert frozen_distribution.cdf(time()).shape == ()
    assert frozen_distribution.pdf(time()).shape == ()
    assert frozen_distribution.ppf(probability()).shape == ()
    assert frozen_distribution.rvs(1, seed=21).shape == ()
    assert frozen_distribution.ichf(probability()).shape == ()
    assert frozen_distribution.isf(probability()).shape == ()
    assert frozen_distribution.isf(0.5) == approx(frozen_distribution.median())
    assert frozen_distribution.dhf(time()).shape == ()

    if fixture_name != "exponential":
        assert frozen_distribution.jac_sf(time()).shape == (2,)
        assert frozen_distribution.jac_hf(time()).shape == (2,)
        assert frozen_distribution.jac_chf(time()).shape == (2,)
        assert frozen_distribution.jac_cdf(time()).shape == (2,)
        assert frozen_distribution.jac_pdf(time()).shape == (2,)
    else:
        assert frozen_distribution.jac_sf(time()).shape == ()
        assert frozen_distribution.jac_hf(time()).shape == ()
        assert frozen_distribution.jac_chf(time()).shape == ()
        assert frozen_distribution.jac_cdf(time()).shape == ()
        assert frozen_distribution.jac_pdf(time()).shape == ()

    n = 10

    assert frozen_distribution.sf(time(n)).shape == (n,)
    assert frozen_distribution.hf(time(n)).shape == (n,)
    assert frozen_distribution.chf(time(n)).shape == (n,)
    assert frozen_distribution.cdf(time(n)).shape == (n,)
    assert frozen_distribution.pdf(time(n)).shape == (n,)
    assert frozen_distribution.ppf(
        probability(
            n,
        )
    ).shape == (n,)
    assert frozen_distribution.rvs((n,), seed=21).shape == (n,)
    assert frozen_distribution.ichf(
        probability(
            n,
        )
    ).shape == (n,)
    assert frozen_distribution.isf(
        probability(
            n,
        )
    ).shape == (n,)
    assert frozen_distribution.isf(np.full((n,), 0.5)) == approx(np.full((n,), frozen_distribution.median()))
    assert frozen_distribution.dhf(
        time(
            n,
        )
    ).shape == (n,)

    if fixture_name != "exponential":
        assert frozen_distribution.jac_sf(
            time(
                n,
            )
        ).shape == (n, 2)
        assert frozen_distribution.jac_hf(
            time(
                n,
            )
        ).shape == (n, 2)
        assert frozen_distribution.jac_chf(
            time(
                n,
            )
        ).shape == (n, 2)
        assert frozen_distribution.jac_cdf(
            time(
                n,
            )
        ).shape == (n, 2)
        assert frozen_distribution.jac_pdf(
            time(
                n,
            )
        ).shape == (n, 2)
    else:
        assert frozen_distribution.jac_sf(
            time(
                n,
            )
        ).shape == (n, 1)
        assert frozen_distribution.jac_hf(
            time(
                n,
            )
        ).shape == (n, 1)
        assert frozen_distribution.jac_chf(
            time(
                n,
            )
        ).shape == (n, 1)
        assert frozen_distribution.jac_cdf(
            time(
                n,
            )
        ).shape == (n, 1)
        assert frozen_distribution.jac_pdf(
            time(
                n,
            )
        ).shape == (n, 1)

    m = 3

    assert frozen_distribution.sf(time(m, 1)).shape == (m, 1)
    assert frozen_distribution.hf(time(m, 1)).shape == (m, 1)
    assert frozen_distribution.chf(time(m, 1)).shape == (m, 1)
    assert frozen_distribution.cdf(time(m, 1)).shape == (m, 1)
    assert frozen_distribution.pdf(time(m, 1)).shape == (m, 1)
    assert frozen_distribution.ppf(probability(m, 1)).shape == (m, 1)
    assert frozen_distribution.rvs((m, 1), seed=21).shape == (m, 1)
    assert frozen_distribution.ichf(probability(m, 1)).shape == (m, 1)
    assert frozen_distribution.isf(probability(m, 1)).shape == (m, 1)
    assert frozen_distribution.isf(np.full((m, 1), 0.5)) == approx(np.full((m, 1), frozen_distribution.median()))
    assert frozen_distribution.dhf(time(m, 1)).shape == (m, 1)

    if fixture_name != "exponential":
        assert frozen_distribution.jac_sf(time(m, 1)).shape == (m, 2)
        assert frozen_distribution.jac_hf(time(m, 1)).shape == (m, 2)
        assert frozen_distribution.jac_chf(time(m, 1)).shape == (m, 2)
        assert frozen_distribution.jac_cdf(time(m, 1)).shape == (m, 2)
        assert frozen_distribution.jac_pdf(time(m, 1)).shape == (m, 2)
    else:
        assert frozen_distribution.jac_sf(time(m, 1)).shape == (m, 1)
        assert frozen_distribution.jac_hf(time(m, 1)).shape == (m, 1)
        assert frozen_distribution.jac_chf(time(m, 1)).shape == (m, 1)
        assert frozen_distribution.jac_cdf(time(m, 1)).shape == (m, 1)
        assert frozen_distribution.jac_pdf(time(m, 1)).shape == (m, 1)

    assert frozen_distribution.sf(time(m, n)).shape == (m, n)
    assert frozen_distribution.hf(time(m, n)).shape == (m, n)
    assert frozen_distribution.chf(time(m, n)).shape == (m, n)
    assert frozen_distribution.cdf(time(m, n)).shape == (m, n)
    assert frozen_distribution.pdf(time(m, n)).shape == (m, n)
    assert frozen_distribution.ppf(probability(m, n)).shape == (m, n)
    assert frozen_distribution.rvs((m, n), seed=21).shape == (m, n)
    assert frozen_distribution.ichf(probability(m, n)).shape == (m, n)
    assert frozen_distribution.isf(probability(m, n)).shape == (m, n)
    assert frozen_distribution.isf(np.full((m, n), 0.5)) == approx(np.full((m, n), frozen_distribution.median()))
    assert frozen_distribution.dhf(time(m, n)).shape == (m, n)
    with pytest.raises(ValueError) as err:
        frozen_distribution.jac_sf(time(m, n))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
    with pytest.raises(ValueError) as err:
        frozen_distribution.jac_hf(time(m, n))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
    with pytest.raises(ValueError) as err:
        frozen_distribution.jac_chf(time(m, n))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
    with pytest.raises(ValueError) as err:
        frozen_distribution.jac_pdf(time(m, n))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)
    with pytest.raises(ValueError) as err:
        frozen_distribution.jac_cdf(time(m, n))
    assert "Unexpected time shape. Got (m, n) shape but only (), (n,) or (m, 1) are allowed here" in str(err.value)



def test_frozen_proportional_hazard(proportional_hazard, time, covar, probability):

    n = 10
    m = 3
    frozen_pph = proportional_hazard.freeze(covar(m))


    assert frozen_pph.moment(1).shape == (m, 1)
    assert frozen_pph.moment(2).shape == (m, 1)
    assert frozen_pph.mean().shape == (m, 1)
    assert frozen_pph.var().shape == (m, 1)
    assert frozen_pph.median().shape == (m, 1)

    assert frozen_pph.sf(time()).shape == (m, 1)
    assert frozen_pph.hf(time()).shape == (m, 1)
    assert frozen_pph.chf(time()).shape == (m, 1)
    assert frozen_pph.cdf(time()).shape == (m, 1)
    assert frozen_pph.pdf(time()).shape == (m, 1)
    assert frozen_pph.ppf(probability()).shape == (m, 1)
    assert frozen_pph.rvs(1, seed=21).shape == (m, 1)
    assert frozen_pph.ichf(probability()).shape == (m, 1)
    assert frozen_pph.isf(probability()).shape == (m, 1)
    assert frozen_pph.isf(0.5) == approx(frozen_pph.median())

    assert frozen_pph.sf(time(n)).shape == (m, n)
    assert frozen_pph.hf(time(n)).shape == (m, n)
    assert frozen_pph.chf(time(n)).shape == (m, n)
    assert frozen_pph.cdf(time(n)).shape == (m, n)
    assert frozen_pph.pdf(time(n)).shape == (m, n)
    assert frozen_pph.ppf(
        probability(
            n,
        ),
    ).shape == (m, n)
    assert frozen_pph.ichf(
        probability(
            n,
        ),
    ).shape == (m, n)

    assert frozen_pph.sf(time(m, 1)).shape == (m, 1)
    assert frozen_pph.hf(time(m, 1)).shape == (m, 1)
    assert frozen_pph.chf(time(m, 1)).shape == (m, 1)
    assert frozen_pph.cdf(time(m, 1)).shape == (m, 1)
    assert frozen_pph.pdf(time(m, 1)).shape == (m, 1)
    assert frozen_pph.ppf(probability(m, 1)).shape == (m, 1)
    assert frozen_pph.rvs((m, 1), seed=21).shape == (m, 1)
    assert frozen_pph.ichf(probability(m, 1)).shape == (m, 1)
    assert frozen_pph.isf(probability(m, 1)).shape == (m, 1)
    assert frozen_pph.isf(np.full((m, 1), 0.5)) == approx(frozen_pph.median())

    assert frozen_pph.sf(time(m, n)).shape == (m, n)
    assert frozen_pph.hf(time(m, n)).shape == (m, n)
    assert frozen_pph.chf(time(m, n)).shape == (m, n)
    assert frozen_pph.cdf(time(m, n)).shape == (m, n)
    assert frozen_pph.pdf(time(m, n)).shape == (m, n)
    assert frozen_pph.ppf(probability(m, n)).shape == (m, n)
    assert frozen_pph.rvs((m, n), seed=21).shape == (m, n)
    assert frozen_pph.ichf(probability(m, n)).shape == (m, n)
    assert frozen_pph.isf(probability(m, n)).shape == (m, n)
    assert frozen_pph.isf(np.full((m, n), 0.5)) == approx(np.broadcast_to(frozen_pph.median(), (m, n)))



def test_frozen_aft(aft, time, covar, probability):

    n = 10
    m = 3
    frozen_aft = aft.freeze(covar(m))


    assert frozen_aft.moment(1).shape == (m, 1)
    assert frozen_aft.moment(2).shape == (m, 1)
    assert frozen_aft.mean().shape == (m, 1)
    assert frozen_aft.var().shape == (m, 1)
    assert frozen_aft.median().shape == (m, 1)

    assert frozen_aft.sf(time()).shape == (m, 1)
    assert frozen_aft.hf(time()).shape == (m, 1)
    assert frozen_aft.chf(time()).shape == (m, 1)
    assert frozen_aft.cdf(time()).shape == (m, 1)
    assert frozen_aft.pdf(time()).shape == (m, 1)
    assert frozen_aft.ppf(probability()).shape == (m, 1)
    assert frozen_aft.rvs(1, seed=21).shape == (m, 1)
    assert frozen_aft.ichf(probability()).shape == (m, 1)
    assert frozen_aft.isf(probability()).shape == (m, 1)
    assert frozen_aft.isf(0.5) == approx(frozen_aft.median())

    assert frozen_aft.sf(time(n)).shape == (m, n)
    assert frozen_aft.hf(time(n)).shape == (m, n)
    assert frozen_aft.chf(time(n)).shape == (m, n)
    assert frozen_aft.cdf(time(n)).shape == (m, n)
    assert frozen_aft.pdf(time(n)).shape == (m, n)
    assert frozen_aft.ppf(
        probability(
            n,
        ),
    ).shape == (m, n)
    assert frozen_aft.ichf(
        probability(
            n,
        ),
    ).shape == (m, n)

    assert frozen_aft.sf(time(m, 1)).shape == (m, 1)
    assert frozen_aft.hf(time(m, 1)).shape == (m, 1)
    assert frozen_aft.chf(time(m, 1)).shape == (m, 1)
    assert frozen_aft.cdf(time(m, 1)).shape == (m, 1)
    assert frozen_aft.pdf(time(m, 1)).shape == (m, 1)
    assert frozen_aft.ppf(probability(m, 1)).shape == (m, 1)
    assert frozen_aft.rvs((m, 1), seed=21).shape == (m, 1)
    assert frozen_aft.ichf(probability(m, 1)).shape == (m, 1)
    assert frozen_aft.isf(probability(m, 1)).shape == (m, 1)
    assert frozen_aft.isf(np.full((m, 1), 0.5)) == approx(frozen_aft.median())

    assert frozen_aft.sf(time(m, n)).shape == (m, n)
    assert frozen_aft.hf(time(m, n)).shape == (m, n)
    assert frozen_aft.chf(time(m, n)).shape == (m, n)
    assert frozen_aft.cdf(time(m, n)).shape == (m, n)
    assert frozen_aft.pdf(time(m, n)).shape == (m, n)
    assert frozen_aft.ppf(probability(m, n)).shape == (m, n)
    assert frozen_aft.rvs((m, n), seed=21).shape == (m, n)
    assert frozen_aft.ichf(probability(m, n)).shape == (m, n)
    assert frozen_aft.isf(probability(m, n)).shape == (m, n)
    assert frozen_aft.isf(np.full((m, n), 0.5)) == approx(np.broadcast_to(frozen_aft.median(), (m, n)))

