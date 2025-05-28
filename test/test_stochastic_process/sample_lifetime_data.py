from relife.lifetime_model import Weibull, Exponential, Gompertz, Gamma, LogLogistic
from relife.stochastic_process import RenewalProcess
import numpy as np

DISTRIBUTION_INSTANCES = [
    Exponential(0.00795203),
    Weibull(3.46597395, 0.01227849),
    Gompertz(0.00865741, 0.06062632),
    Gamma(5.3571091, 0.06622822),
    LogLogistic(3.92614064, 0.0133325),
]

for distri in DISTRIBUTION_INSTANCES:
    renewal_process = RenewalProcess(distri)
    expected_params = distri.params.copy()
    q1 = distri.ppf(0.25)
    q3 = distri.ppf(0.75)
    success = 0
    n = 100
    for i in range(n):
        lifetime_data = renewal_process.sample_lifetime_data(10 * q3, t0=q1, size=100)
        try: # for gamma and loglogistic essentially (convergence errors may occcur)
            distri.fit_from_lifetime_data(lifetime_data)
        except RuntimeError:
            continue
        ic = distri.fitting_results.IC
        if np.all(distri.params.reshape(-1,1) >= ic[:, [0]]) and np.all(distri.params.reshape(-1,1) <= ic[:, [1]]):
            success += 1
    if success >= 0.95*n:
        print(type(distri).__name__, "OK")
    else:
        print(type(distri).__name__, "FAILED")
