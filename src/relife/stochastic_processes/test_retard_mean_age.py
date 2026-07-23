import numpy as np
from relife.lifetime_models import Weibull
from relife.stochastic_processes import RenewalProcess

def test_mean_age_sample():
    """ Test sans a0- Cas simple """
    model = Weibull(2,0.05)
    process = RenewalProcess(lifetime_model=model)
    timeline, values = process.mean_age(tf=100, nb_steps = 50)
    assert values[0] == 0.0
    
def test_mean_age_delayed():
    """ Test avec a0- Cas retardé """
    model = Weibull(2,0.05)
    process = RenewalProcess(lifetime_model=model)
    timeline, values = process.mean_age(tf=100, nb_steps = 50, a0 = 5)
    assert values[0] >= 0.0
    