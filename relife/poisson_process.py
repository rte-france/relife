from __future__ import annotations

import numpy as np

from typing import Tuple


from scipy.optimize import Bounds
from dataclasses import dataclass
from .data import LifetimeData, CountData
from .parametric import ParametricHazardFunctions
from .nonparametric import NelsonAalen
from .utils import plot



# Où mettre Nelson Aalen pour NHPP ?
# soit dans NonHomogeneousPoissonProcess, soit dans nonparametric.py ?

@dataclass
#class NonHomogeneousPoissonProcessData(CountData):
class NonHomogeneousPoissonProcessData:
    # Si j'herite de count data il faut que 
    #T deviennent un np.ndarray et il y la fin de de l'observation de l'asset (perso je prefere parler de processus)
    #A priori je n'ai pas besoin de n_indices et n_samples
    #Je ne suis pas sur de bien comprendre indices et samples mais potentiellement c'est exactement ce dont j'ai besoin samples doit etre indices des times et indices l'indice des asset
    #Il faudra rajouter un T0 np:ndarray
    #Version sans count data
    times: np.ndarray #: Ages of all events for every assets.
    t0: np.ndarray #: Initial observations age of the assets.
    tf: np.ndarray #: Final observations age of the assets.
    times_indices: np.ndarray #: Times' Events asset corresponding indices (in the same order than times).
    assets_indices: np.ndarray #: Assets indices
    args: Tuple[np.ndarray] = ()  #: Extra arguments required by the NHPP model.

    def to_lifetime_data(self):
        entry = np.concatenate((self.times,self.t0))
        time = np.concatenate((self.times,self.tf))
        indices = np.concatenate((self.times_indices,self.assets_indices))
        event = np.concatenate((np.ones(len(self.times_indices)),np.zeros(len(self.assets_indices))))


        entry_ordered_ind = np.lexsort((entry,indices))# Sort by indices, then by entry
        time_ordered_ind = np.lexsort((time,indices))

        entry = entry[entry_ordered_ind]
        time = time[time_ordered_ind]
        event = event[time_ordered_ind]
        return LifetimeData(time,event,entry)



@dataclass
class NonHomogeneousPoissonProcess(ParametricHazardFunctions):

    model: ParametricHazardFunctions

    #Equivalent à 
    #def __init__(self, model:ParametricHazardFunctions)
    #    self.model = model

    #nhpp = NonHomogeneousPoissonProcess(Weibull())
    #nhpp.fit(times, ...)
    #print(nhpp)
    #> NonHomogeneousPoissonProcess(model=Weibull(2.02252554, 0.0455655))

    @property
    def params(self) -> np.ndarray:
        return self.model.params

    @property
    def n_params(self) -> int:
        return self.model.n_params

    @property
    def _param_bounds(self) -> Bounds:
        return self.model._param_bounds

    def _set_params(self, params: np.ndarray) -> None:
        return self.model._set_params(params)

    def _init_params(self, data: LifetimeData) -> np.ndarray:
        return self.model._init_params(data)

    def _chf(self, params: np.ndarray, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.model._chf(params, t, *args)

    def _hf(self, params: np.ndarray, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.model._hf(params, t, *args)

    def _dhf(self, params: np.ndarray, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.model._dhf(params, t, *args)

    def _jac_chf(
        self, params: np.ndarray, t: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        return self.model._jac_chf(params, t, *args)

    def _jac_hf(
        self, params: np.ndarray, t: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        return self.model._jac_hf(params, t, *args)

    def _ichf(self, params: np.ndarray, x: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.model._ichf(params, t, *args)

    def fit(self, times, times_indices=None, assets_indices=None, t0=None, tf=None, params0: np.ndarray = None,
        method: str = None,**kwargs):
        
        data = NonHomogeneousPoissonProcessData(times=times, times_indices=times_indices, assets_indices=assets_indices, t0=t0, tf=tf).to_lifetime_data()
        self._fit(data, params0, method=method, **kwargs)
        return self

    def plot(
        self,
        timeline: np.ndarray = None,
        args: Tuple[np.ndarray] = (),
        alpha_ci: float = 0.05,
        fname: str = "chf",
        **kwargs,
    ) -> None:
        flist = ["chf", "hf"]
        if fname not in flist:
            raise ValueError(
                "Function name '{}' is not supported for plotting, `fname` must be in {}".format(
                    fname, flist
                )
            )
        return self.model.plot(timeline,args,alpha_ci,fname,**kwargs)
    
    #def plot(
    #    self,
    #    timeline: np.ndarray = None,
    #    args: Tuple[np.ndarray] = (),
    #    alpha_ci: float = 0.05,
    #    fname: str = "sf",
    #    **kwargs,
    #) -> None:

    def sample(self, t0, tf):
        pass


class NonParametricCumulativeIntensityFunction(NelsonAalen):
    #ATTENTION : Il faudra overwrite les docstring !!
    r"""Non-Parametric Cumulative Intensity Function

    """
    def fit(
        self, times: np.ndarray, times_indices: np.ndarray=None, assets_indices: np.ndarray=None, t0: np.ndarray=None, tf: np.ndarray=None
        ) -> NonParametricCumulativeIntensityFunction:
        data = NonHomogeneousPoissonProcessData(times=times, times_indices=times_indices, assets_indices=assets_indices, t0=t0, tf=tf).to_lifetime_data()
        return super().fit(time = data.time, event = data.event, entry = data.entry)

    def plot(self, alpha_ci: float = 0.05, **kwargs: np.ndarray) -> None:
        r"""Plot the Nelson-Aalen estimator of the cumulative hazard function.
        """
        label = kwargs.pop("label", "Non-parametric cumulative intensity function")
        return super().plot(alpha_ci,label=label,**kwargs)

