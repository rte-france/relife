import numpy as np
import pytest

from relife2 import ECDF, KaplanMeier, NelsonAalen, Turnbull
from relife2 import Weibull
from relife2.utils.dataset import load_power_transformer, load_input_turnbull


@pytest.fixture
def data():
    return load_power_transformer()


@pytest.fixture
def data_turnbull():
    return load_input_turnbull()


@pytest.fixture(
    scope="module",
    params=[Weibull(7.531, 0.037)],
)
def weibull_model(request):
    return request.param


# test functions


def test_fit_ecdf_kaplan_meier(data):
    time, event, entry = data
    ecdf = ECDF()
    ecdf.fit(time)
    km = (
        KaplanMeier()
    )  ## ATTENTION faut tester sans entry ni rc_indic pr être cohérent avec ECDF,
    km.fit(time)
    _km = (
        KaplanMeier()
    )  # mais tester l'estimate avec rc_indic qd mm pour verifier que tt fonctionne shape wise
    _km.fit(time, event, entry)
    _na = NelsonAalen()
    _na.fit(time=time, entry=entry)

    assert ecdf.estimates["sf"].values == pytest.approx(
        km.estimates["sf"].values, rel=1e-4
    )
    assert np.isclose(ecdf.sf(91), 0.00060606)
    assert np.isclose(km.sf(0.1), 0.99818182)


def test_turnbull(data_turnbull, weibull_model):
    """
    Test the Turnbull estimator. Data contains also exact observations.

    Explications :
        - Lors de la création de data_turnbull, on s'est basé sur Weibull(c=7.531, rate=0.037)
    pour déterminer si un interval contenait une defaillance ou non.

        - Division de tb.timeline par 6 : On avait multiplié les temps de Weibull(c=7.531, rate=0.037)
    par 6 pour avoir des valeurs cohérente avec les valeurs des intervales de visite de poste.

    """
    time = np.array(data_turnbull[:-1]).T
    tb = Turnbull(lowmem=True)
    tb.fit(time, entry=data_turnbull[-1])
    t = tb.estimates["sf"].timeline / 6
    assert np.isclose(tb.estimates["sf"].values, weibull_model.sf(t), atol=0.02).all()
    assert np.isclose(tb.sf(100), 0.97565265)
