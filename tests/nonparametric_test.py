import numpy as np
import pytest

from datasets import load_power_transformer, load_input_turnbull
from relife2 import ECDF, KaplanMeier, NelsonAalen, Turnbull, Weibull  # TODO test it


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
    rc_indic = 1 - event
    ecdf = ECDF(time=time)
    km = KaplanMeier(
        time=time
    )  ## ATTENTION faut tester sans entry ni rc_indic pr être cohérent avec ECDF,

    _km = KaplanMeier(
        time=time, entry=entry, rc_indicators=rc_indic
    )  # mais tester l'estimate avec rc_indic qd mm pour verifier que tt fonctionne shape wise
    _na = NelsonAalen(time=time, entry=entry, rc_indicators=rc_indic)

    assert ecdf.sf.values == pytest.approx(km.sf.values, rel=1e-4)


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
    tb = Turnbull(time, entry=data_turnbull[-1], lowmem=True)
    t = tb.sf.timeline / 6
    assert np.isclose(tb.sf.values, weibull_model.sf(t), atol=0.02).all()
