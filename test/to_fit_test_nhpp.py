import numpy as np
from matplotlib import pyplot as plt

from relife.process import NonHomogeneousPoissonProcess
from relife.models import Weibull
from relife.data import nhpp_data_factory

# a0 = np.array([1.0, 3.0, 6.0])
# af = np.array([5.0, 8.0, 15.0])
# ages = np.array([2.0, 4.0, 12.0, 13.0, 14.0])
# assets = np.array([0, 0, 2, 2, 2], dtype=np.int64)
# time, event, entry = nhpp_lifetime_data_factory(a0, af, ages, assets)

model = Weibull(7, 0.05)
print("true params", model.params)
print("expected lifetime", model.mean())
nhpp = NonHomogeneousPoissonProcess(model)
true_params = nhpp.params
# cumulative_intensity(end_time) : nb moyen de pannes sur l'intervalle [0, end_time]

# error if end_time 20
# error if 1, 1
# error if 20, 1, 35 (ok pour 25)
# stop if duration exceed
nhpp_data = nhpp.sample(20, 1, end_time=35)

a0, af, ages, assets = nhpp_data.to_fit()
print("nb data :", len(ages))
nhpp2 = NonHomogeneousPoissonProcess(Weibull())
nhpp2.fit(a0, af, ages, assets, inplace=True)
estimated_params = nhpp2.params
print("estimated params", nhpp2.params)


from relife.policies import NHPPAgeReplacementPolicy


model = Weibull(7, 0.05)
model.plot.sf()
plt.show()


for cr in np.array(
    [
        # 1e-3,
        # 1e-2,
        1e-1,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        20.0,
        100.0,
        1000.0,
        1e4,
        1e5,
    ]
):
    policy = NHPPAgeReplacementPolicy(Weibull(7, 0.05), 5.0, cr).fit()
    print(policy.ar)
