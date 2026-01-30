
from pathlib import Path

import pandas as pd
import numpy as np

from relife.lifetime_model import Cox
from relife.statistical_tests import cox_snell_residuals


# Données chaines d'isolateur
relife_csv_datapath = Path(r"D:\Projets\RTE\ReLife\relife\relife\data\csv")
time, event, entry, *args = np.loadtxt(relife_csv_datapath / "insulator_string.csv", delimiter=",", skiprows=1,
                                       unpack=True)
covar = np.column_stack(args)

# Into df
data = pd.DataFrame({"time": time, "event": event, "entry": entry})
covar = pd.DataFrame(covar)
covar.columns = [f"covar_{i}" for i in range(covar.shape[1])]
data = pd.concat([data, covar], axis=1)

# Relife model fit
re_model = Cox()
re_model.fit(
    time=data["time"].values,
    covar=data.filter(regex="covar").values,
    event=data["event"].values,
)
print(re_model.fitting_results)

# Relife proportionality_effect test
#likelihood_ratio_test(re_model, model_init_kwargs={}, c=np.array([1, 1, 0]), optimizer_options=None, seed=1)
print(cox_snell_residuals(
    re_model
))