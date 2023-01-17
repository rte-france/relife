from __future__ import annotations

import numpy as np

from typing import Tuple


from scipy.optimize import Bounds
from dataclasses import dataclass
from .data import LifetimeData, CountData
from .parametric import ParametricHazardFunctions
from .nonparametric import NelsonAalen
from .utils import plot, args_take



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
    #args = ()  #: Extra arguments required by the NHPP model.



    def to_lifetime_data(self):

        ar  = np.column_stack((self.times_indices,self.times))
        if len(np.vstack({tuple(e) for e in ar})) < len(ar):
            raise ValueError("A process can have only one event at the same time")

        entry = np.concatenate((self.times,self.t0))
        time = np.concatenate((self.times,self.tf))
        indices = np.concatenate((self.times_indices,self.assets_indices))
        event = np.concatenate((np.ones(len(self.times_indices)),np.zeros(len(self.assets_indices))))


        entry_ordered_ind = np.lexsort((entry,indices))# Sort by indices, then by entry
        time_ordered_ind = np.lexsort((time,indices))

        entry = entry[entry_ordered_ind]
        time = time[time_ordered_ind]
        event = event[time_ordered_ind]
        


        #args_LD_format = args_take(indices[entry_ordered_ind].astype(int), self.args)

        condition = ((entry==time)&(event==0))

        #args = (covar, *args) if covar is not None else args
        if type(self.args) is not tuple:
            num_to_repeat = np.histogram(self.times_indices,np.insert(self.assets_indices,self.assets_indices.shape[0],self.assets_indices.shape[0]))[0]+1
            args_LD_format = np.repeat(self.args,num_to_repeat,axis=0)
            if any(condition):
                args_LD_format = args_LD_format[~condition] 
            args_LD_format = [args_LD_format]
        else:
            args_LD_format = self.args




        #args_LD_format = [self.args] if type(self.args) is not tuple else self.args
        #args_LD_format = args_take(indices[entry_ordered_ind].astype(int), *args_LD_format)

        

        if any(condition):
            entry = entry[~condition]
            time = time[~condition]
            event = event[~condition]

        return LifetimeData(time,event,entry,args_LD_format)
        #return LifetimeData(time,event,entry,args_LD_format)
        #return LifetimeData(time,event,entry,*args_take(indices[entry_ordered_ind].astype(int), *self.args)) if type(self.args) is tuple else LifetimeData(time,event,entry,*args_take(indices[entry_ordered_ind].astype(int), self.args))
        #return LifetimeData(time,event,entry,*args_take(indices[entry_ordered_ind].astype(int), *self.args)) if not tuple(map(tuple, self.args)) else LifetimeData(time,event,entry,*args_take(indices[entry_ordered_ind].astype(int), self.args))
              


        #np.take(arg, indices, axis=-2)
        #if np.ndim(arg) > 0
        #else np.tile(arg, (np.size(indices), 1))


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
        method: str = None, args: Tuple[np.ndarray] = (), **kwargs):
        
        data = NonHomogeneousPoissonProcessData(times=times, times_indices=times_indices, assets_indices=assets_indices, t0=t0, tf=tf,args=args).to_lifetime_data()
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


    def sample(number_of_sample,model_nhpp,T0,TF):
        event_per_sample = 50
        n_while = 1
        expo_times = np.random.exponential(scale=1.0, size=(number_of_sample, event_per_sample))
    
        cs_expo_times = np.cumsum(expo_times,axis=1)
        max_expo = np.max(cs_expo_times,axis=1)
    
        nhpp_times = model_nhpp.ichf(cs_expo_times)
        maxNHPPtime = np.max(nhpp_times,axis=1)

        while any(maxNHPPtime.reshape(-1,1)<TF):

            n_while = n_while + 1      
            expo_times = np.random.exponential(scale=1.0, size=(number_of_sample, event_per_sample))
            expo_times[:,0] = expo_times[:,0]+max_expo
            to_append = np.cumsum(expo_times,axis=1)
            cs_expo_times = np.append(cs_expo_times,to_append,axis=1)
            max_expo = np.max(cs_expo_times,axis=1)
            nhpp_times = model_nhpp.ichf(cs_expo_times)
            maxNHPPtime = np.max(nhpp_times,axis=1)#à comparer avec le TF en vue d'une boucle while


        numbers = np.hstack([np.arange(0, number_of_sample)[:, None]]*event_per_sample*n_while)
        mask = (nhpp_times < TF) & (nhpp_times > T0)
        nhpp_times = nhpp_times[mask]
        no_observation = np.all(np.invert(mask),axis=1)


        numbers = numbers[mask]
        sample_informations = np.insert(np.concatenate((T0,TF),axis=1), 0, np.arange(0, number_of_sample), axis=1)

        return(sample_informations,numbers,nhpp_times)

    #def sample(self, t0, tf):
    #    pass


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

