from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sc
from scipy import integrate
from scipy.optimize import minimize, newton
from scipy.stats import gamma, loggamma

from relife.model import AbsolutelyContinuousLifetimeModel
from .utils import moore_jac_uppergamma_c


# matplotlib.use('Qt5Agg', force=True)


@dataclass
class GammaProcessData:
    inspection_times: np.array
    deterioration_measurements: np.array
    ids: np.array = None
    censor: float = 0  # niveau de la censure par ex : precision de l'instrument de mesure
    increments: np.array = None
    log_increments: np.array = None
    _event: np.array = None

    def __post_init__(self):
        # TODO: parse.data() beaucoup trop longue à s'exécuter: énorme bottleneck
        if self.ids is None:
            self.ids = np.ones(len(self.inspection_times), dtype=int)
        self.unique_ids = np.unique(self.ids)
        self.increments = np.diff(self.deterioration_measurements)
        self.parse_data()

    def plot(self):
        for i in self.unique_ids:
            plt.plot(self.inspection_times[self.ids == i], self.deterioration_measurements[self.ids == i],
                     alpha=0.5)
        plt.xlabel('t')
        plt.ylabel('Deterioration')
        plt.show()

    def parse_data(self) -> None:

        if np.size(self.ids) != np.size(self.inspection_times):
            raise ValueError("'inspection_times' and 'ids' must have the same length")

        if np.size(self.inspection_times) != np.size(self.deterioration_measurements):
            raise ValueError("'inspection_times' and 'deterioration_measurements' must have the same length")

        # TODO: à revoir
        # check if user specified an 'inspection_times' = 0 for some 'ids', and accordingly, check if corresponding
        # 'deterioration_measurements' is zero, otherwise raise an error. If 'inspection_times' and
        # 'deterioration_measurements' are well specified, removes all instances of 'inspection_times' = 0.

        check_initial_inspection_times_per_id_is_zero = np.array(
            [any(self.inspection_times[self.ids == i] == 0) for i in self.unique_ids]
        )

        invalid_id_starting_values = self.unique_ids[np.array(
            [(self.deterioration_measurements[self.ids == i][0] > 0)
             & (self.inspection_times[self.ids == i][0] == 0) for i in self.unique_ids]
        )]

        if len(invalid_id_starting_values) > 0:
            raise ValueError(
                f"Invalid starting values for 'inspection_times' and "
                f"'deterioration_measurements' for ids {invalid_id_starting_values}")

        else:
            ind = np.where(check_initial_inspection_times_per_id_is_zero)[0]
            ind = self.unique_ids[ind]
            ind = np.array([np.where(self.ids == i)[0][0] for i in ind])

            if len(ind) > 0:
                self.inspection_times = np.delete(self.inspection_times, ind)
                self.deterioration_measurements = np.delete(self.deterioration_measurements, ind)
                self.ids = np.delete(self.ids, ind)

        # Inserting 'inspection_times' = 0, 'deterioration_measurements' = 0 for all 'ids', and calculating
        # 'increments' with the convention that the increments corresponding to 'inspection_times' = 0 is also 0.
        first_id_index = [np.where(self.ids == i)[0][0] for i in np.unique(self.ids)]
        self.ids = np.insert(self.ids, first_id_index, np.unique(self.ids))

        self.inspection_times = np.insert(self.inspection_times, first_id_index, 0)
        self.deterioration_measurements = np.insert(self.deterioration_measurements, first_id_index, 0)

        self.increments = np.concatenate(
            [np.diff(self.deterioration_measurements[self.ids == i]) for i in self.unique_ids])
        self.increments = np.insert(self.increments, first_id_index, 0)

        # Some more tests
        if any(self.inspection_times < 0):
            raise ValueError("'inspection_times' must be positive")

        if np.size(self.inspection_times) == 1:
            raise ValueError(
                "'inspection_times' and 'deterioration_measurements' must contain at least two data points")

        # check if 'inspections_times' are increasing for each 'ids'
        check_inspection_times_per_id = [any(np.diff(self.inspection_times[self.ids == i]) <= 0) for i in
                                         self.unique_ids]
        if any(check_inspection_times_per_id):
            incorrect_ids = self.unique_ids[np.where(check_inspection_times_per_id)[0]]
            raise ValueError(f"'ids' {incorrect_ids} have non increasing 'inspection_times'")

        # check if 'deterioration_measurements' are increasing for each 'ids'
        check_deterioration_measurements_per_id = [any(np.diff(self.deterioration_measurements[self.ids == i]) < 0) for
                                                   i in self.unique_ids]
        if any(check_deterioration_measurements_per_id):
            incorrect_ids = self.unique_ids[np.where(check_deterioration_measurements_per_id)[0]]
            raise ValueError(f"'ids' {incorrect_ids} have non increasing 'deterioration_measurements'")

        self.increments[self.increments <= self.censor] = 0
        self._event = self.increments == 0

        if self.log_increments is None:
            self.log_increments = np.log(self.increments, where=self.increments != 0)


@dataclass
class GammaProcess(AbsolutelyContinuousLifetimeModel):
    r0: float = None
    l0: float = None
    rate: float = None
    shape_rate: float = None
    shape_power: float = None

    @property
    def model_params(self) -> np.ndarray:
        return np.array([self.shape_rate, self.shape_power, self.rate, self.r0, self.l0])

    @staticmethod
    def _shape_function(params, t):
        return params[0] * t ** params[1]

    def shape_function(self, t):
        return self._shape_function(self.model_params, t)

    def sample(self, inspection_times, nb_sample=1):
        n = len(inspection_times) - 1
        h = np.diff(inspection_times)
        shape = np.repeat((self.shape_function(inspection_times[:-1] + h)
                           - self.shape_function(inspection_times[:-1])).reshape(1, -1),
                          nb_sample,
                          axis=0)
        # increments = np.random.gamma(shape, 1 / self.rate, (nb_sample, n))

        log_increments = loggamma.rvs(c=shape, size=(nb_sample, n)) - np.log(
            self.rate)  # generate loggamma distribution to avoid underflow. Useful for optimizing log likelihood

        increments = np.exp(log_increments)
        deterioration_measurements = np.cumsum(increments, axis=1)
        ids = np.repeat(np.arange(nb_sample), n)
        log_increments = np.concatenate((np.zeros(nb_sample)[:, np.newaxis], log_increments), axis=1).ravel()
        gp_data = GammaProcessData(np.tile(inspection_times[1:], nb_sample),
                                   deterioration_measurements.ravel(),
                                   ids,
                                   log_increments=log_increments)

        return gp_data

    def plot(self, inspection_times):
        plt.plot(inspection_times, self.shape_function(inspection_times), linewidth=3)
        plt.xlabel('t')
        plt.title('Gamma process shape function: $v(t) = ct^b$')
        plt.show()

    def support_upper_bound(self, *args: np.ndarray) -> float:
        return np.inf

    def sf(self, t):
        return sc.gammainc(self.shape_function(t), (self.r0 - self.l0) * self.rate)

    def conditional_sf(self, t, s, x):
        return sc.gammainc(self.shape_function(t) - self.shape_function(s), (x - self.l0) * self.rate)

    def _sf(self, params, t):
        shape_rate, shape_power, rate = params
        return sc.gammainc(self._shape_function(params, t), (self.r0 - self.l0) * rate)

    def pdf(self, t):
        ind0 = np.where(t == 0)[0]
        non0_times = np.delete(t, ind0)
        res = -self.shape_power / non0_times * self.shape_function(non0_times) \
               * moore_jac_uppergamma_c(P=self.shape_function(non0_times), x=(self.r0 - self.l0) * self.rate)

        if self.shape_power == 1:
            return np.insert(res, ind0, -self.shape_rate * sc.expi(-(self.r0 - self.l0) * self.rate))
        else:
            return np.insert(res, ind0, 0)

    def hf(self, t):
        return self.pdf(t) / self.sf(t)

    def chf(self, t):
        return -np.log(self.sf(t))

    # TODO: ichf often fails to convergence: changer améliorer l'initialisation
    def ichf(self, v: np.ndarray, *args: np.ndarray) -> np.ndarray:
        initial_guess = np.ones_like(v)
        return newton(func=lambda t: self.chf(t) - v, x0=initial_guess, fprime=lambda t: self.hf(t))

    def isf(self, p: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.ichf(-np.log(p), *args)

    def _negative_log_likelihood(self, params, data):
        shape_rate, shape_power, rate = params
        censor = data.censor

        negative_log_likelihood_ids = []
        for i in data.unique_ids:
            inspection_times_id = data.inspection_times[data.ids == i]
            increments_id = data.increments[data.ids == i][1:]
            log_increments_id = data.log_increments[data.ids == i][1:]
            event_id = data._event[data.ids == i][1:]

            # contribution of uncensored measurements to likelihood
            exact_contribution = - np.sum(
                np.diff(self._shape_function(params, inspection_times_id))[~event_id] * np.log(rate)
                - np.log(sc.gamma(shape_rate * (np.diff(inspection_times_id ** shape_power))[~event_id]))
                + (shape_rate * (np.diff(inspection_times_id ** shape_power))[~event_id] - 1)
                * log_increments_id[~event_id] - rate * increments_id[~event_id]
            )

            # contributution of censored measurements to likelihood
            if censor != 0:
                censored_contribution = - np.sum(
                    np.log(gamma.cdf(censor, a=np.diff(self._shape_function(params, inspection_times_id))[event_id],
                                     scale=1 / rate))
                )
            else:
                censored_contribution = 0

            negative_log_likelihood_ids.append(exact_contribution + censored_contribution)

        return np.sum(negative_log_likelihood_ids)

    def _jac_negative_log_likelihood(self, params: np.ndarray, data: GammaProcessData) -> np.ndarray:
        pass

    @staticmethod
    def _method_of_moments(shape_power, data):

        moment1, moment2, moment3, moment2_scaling, moment3_scaling = [], [], [], [], []
        for i in data.unique_ids:
            wi_id = np.diff(data.inspection_times[data.ids == i] ** shape_power)
            zi_id = np.diff(data.deterioration_measurements[data.ids == i])
            zbar_id = np.sum(zi_id) / np.sum(wi_id)

            moment1.append(
                zbar_id
            )
            moment2.append(
                np.sum((zi_id - zbar_id * wi_id) ** 2)
            )
            moment3.append(
                np.sum((zi_id - zbar_id * wi_id) ** 3)
            )
            moment2_scaling.append(
                np.sum(wi_id) - np.sum(wi_id ** 2) / np.sum(wi_id)
            )
            moment3_scaling.append(
                np.sum(wi_id) + 2 * np.sum(wi_id ** 3) / np.sum(wi_id) ** 2 - 3 * np.sum(wi_id ** 2) / np.sum(wi_id)
            )

        moment1 = np.mean(moment1)
        moment2 = np.mean(moment2)
        moment3 = np.mean(moment3)
        moment2_scaling = np.mean(moment2_scaling)
        moment3_scaling = np.mean(moment3_scaling)

        rate = moment1 * moment2_scaling / moment2
        shape_rate = moment1 * rate
        return np.abs(
            2 * shape_rate / rate ** 3 * moment3_scaling - moment3) ** 2  # la fonction carrée est une pénalité

    @staticmethod
    def return_param(shape_power, data):
        moment1, moment2, moment3, moment2_scaling, moment3_scaling = [], [], [], [], []
        for i in data.unique_ids:
            wi_id = np.diff(data.inspection_times[data.ids == i] ** shape_power)
            zi_id = np.diff(data.deterioration_measurements[data.ids == i])
            zbar_id = np.sum(zi_id) / np.sum(wi_id)

            moment1.append(
                zbar_id
            )
            moment2.append(
                np.sum((zi_id - zbar_id * wi_id) ** 2)
            )
            moment3.append(
                np.sum((zi_id - zbar_id * wi_id) ** 3)
            )
            moment2_scaling.append(
                np.sum(wi_id) - np.sum(wi_id ** 2) / np.sum(wi_id)
            )
            moment3_scaling.append(
                np.sum(wi_id) + 2 * np.sum(wi_id ** 3) / np.sum(wi_id) ** 2 - 3 * np.sum(wi_id ** 2) / np.sum(wi_id)
            )

        moment1 = np.mean(moment1)
        moment2 = np.mean(moment2)
        moment2_scaling = np.mean(moment2_scaling)

        rate = moment1 * moment2_scaling / moment2
        shape_rate = rate * moment1
        return shape_rate, shape_power, rate

    def fit(self, inspection_times, deterioration_measurements, ids, log_increments=None, censor=0):
        data = GammaProcessData(inspection_times=inspection_times,
                                deterioration_measurements=deterioration_measurements,
                                ids=ids,
                                log_increments=log_increments,
                                censor=censor)

        # MOM
        # if any(data.log_increments[data.inspection_times != 0] == 0) or any(
        #         data.increments[data.inspection_times != 0] == 0):
        if False:
            opt = minimize(
                fun=self._method_of_moments,
                x0=np.array([1]),
                args=(data,),
                method='Nelder-Mead',
                bounds=((0.1, 3),),
                options={'maxiter': 1000},
            )
            print("MOM")
            return self.return_param(opt.x[0], data)

        ## Likelihood
        else:
            opt = minimize(
                fun=self._negative_log_likelihood,
                x0=np.array([1, 1, 1]),
                args=(data,),
                method='Nelder-Mead',
                bounds=((1e-3, np.inf),
                        (1e-3, np.inf),
                        (1e-3, np.inf))
            )

            print("Likelihood")
            return opt.x

    def resistance_sample(self, inspection_times, nb_sample=1):
        data = self.sample(inspection_times=inspection_times, nb_sample=nb_sample)
        data.deterioration_measurements = self.r0 - data.deterioration_measurements

        if any([np.min(data.deterioration_measurements[data.ids == i]) > self.l0 for i in data.unique_ids]):
            raise Exception("Some assets did not hit load threshold. Consider increasing the time horizon.")
        ind_failure = [np.argmax(data.deterioration_measurements[data.ids == i] < self.l0) for i in data.unique_ids]
        failure_times = inspection_times[ind_failure]
        return failure_times

    def resistance_plot(self, inspection_times, nb_sample=1):
        data = self.sample(inspection_times=inspection_times, nb_sample=nb_sample)
        data.deterioration_measurements = self.r0 - data.deterioration_measurements

        if any([np.min(data.deterioration_measurements[data.ids == i]) > self.l0 for i in data.unique_ids]):
            raise Exception("Some assets did not hit load threshold. Consider increasing the time horizon.")

        ind_failure = [np.argmax(data.deterioration_measurements[data.ids == i] < self.l0) for i in data.unique_ids]
        failure_times = inspection_times[ind_failure]

        nb_sample_max = nb_sample
        if nb_sample > 100:
            nb_sample_max = 100

        fig, ax1 = plt.subplots(dpi=160)
        color = 'tab:grey'
        ax1.set_xlabel('t')
        ax1.set_ylabel('Asset resistance', color=color)

        for i in range(nb_sample_max):
            inspection_times_nb = inspection_times[0:ind_failure[i] + 1]
            resistance_nb = data.deterioration_measurements[data.ids == i][0:ind_failure[i] + 1]
            resistance_nb[-1] = self.l0
            if i == 0:
                ax1.plot(inspection_times_nb, resistance_nb, color=color, alpha=0.2, label='Asset resistance')
            else:
                ax1.plot(inspection_times_nb, resistance_nb, color=color, alpha=0.2)

        b = 1.025
        a = 1 - self.r0 * (b - 1) / self.l0
        plt.ylim(a * self.l0, b * self.r0)
        # TODO: ne pas enlever la première valeur et intégrer dans self.pdf la valeur en 0
        ax1.axhline(y=self.r0, color='green', linestyle='--', label='Initial resistance $R_0$')
        ax1.axhline(y=self.l0, color='red', linestyle='--', label='Load threshold $l_0$')
        ind_max = np.argmax(self.r0 - self.shape_function(inspection_times) / self.rate < self.l0)
        t_max = inspection_times[0:ind_max + 1]
        mean_resistance = (self.r0 - self.shape_function(t_max) / self.rate)
        mean_resistance[-1] = self.l0
        ax1.plot(t_max, mean_resistance, color='black', linewidth=2, label='Mean resistance')
        # ax1.legend(loc=2)
        ax1.legend(loc='center left', bbox_to_anchor=(1.2, 0.8), frameon=False)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Density of failure times', color=color)
        ax2.plot(inspection_times[1:], self.pdf(inspection_times[1:]), color=color, linewidth=2,
                 label='True density of failure times')
        ax2.hist(failure_times, color=color, density=True, alpha=0.3, label='Empirical distribution of failure times')
        ax2.tick_params(axis='y', labelcolor=color)
        # ax2.legend(loc=3)
        ax2.legend(loc='center left', bbox_to_anchor=(1.2, 0.6), frameon=False)

        # fig.tight_layout()
        plt.show()

    @staticmethod
    def count_greater(x, inspection_times, resistance_measurements, control_times):
        def interpolated_resistance(t):
            return np.interp(x=t, xp=inspection_times, fp=resistance_measurements)

        return sum(interpolated_resistance(control_times) >= x)

    def replacement_strategy_plot(self, strategy):

        control_frequency, replacement_threshold = strategy
        control_times = np.arange(1, 100) * control_frequency
        # 1) Simulation de la trajectoire d'un PG X(t)
        T_max = 1
        while self.sf(T_max) > 1e-4:
            T_max = T_max + 1
        inspection_times = np.linspace(0, T_max, num=1000)
        data = self.sample(inspection_times=inspection_times, nb_sample=1)

        # 2) Calcul de la résistance R(t) = r0 - X(t)
        resistance_measurements = self.r0 - data.deterioration_measurements
        kl = self.count_greater(replacement_threshold, inspection_times, resistance_measurements, control_times)
        kl0 = self.count_greater(self.l0, inspection_times, resistance_measurements, control_times)
        # 3) Calcul de la durée de vie de l'actif, t_defaillance = inf{t | R(t) < l0}
        ind_defaillance = np.where(resistance_measurements < self.l0)[0][0]
        t_defaillance = inspection_times[ind_defaillance]
        inspection_times = inspection_times[0:ind_defaillance + 1]
        resistance_measurements = resistance_measurements[0:ind_defaillance + 1]
        resistance_measurements[-1] = self.l0
        if len(np.where(control_times <= inspection_times[-1])[0]) == 0:
            inspec_times = np.repeat(control_times[0], 2)
        else:
            inspec_times = control_times[:(np.where(control_times <= inspection_times[-1])[0][-1] + 1)]
        epsilon_l_l0 = kl0 - kl
        if epsilon_l_l0 > 0:
            lifetime = control_times[kl]
            last_ind = np.where(inspection_times <= lifetime)[0][-1]
            resistance_measurements = resistance_measurements[0:last_ind + 1]
            inspection_times = inspection_times[0:last_ind + 1]

        else:
            lifetime = t_defaillance

        plt.plot(inspection_times, resistance_measurements, label='Deterioration process $R(t)$')

        plt.ylim(0, 1.1 * self.r0)
        plt.axhline(y=self.r0, color='green', linestyle='--', label='Initial resistance $R_0$')
        plt.axhline(y=self.l0, color='red', linestyle='--', label='Load threshold $l_0$')
        plt.axhline(y=replacement_threshold, color='black', linestyle='--', label='Replacement threshold $l$')
        plt.axvline(x=inspec_times[0], color='grey', alpha=0.2, linestyle='--', label='Inspection')
        plt.axvline(x=lifetime, color='orange', linestyle='-', label='Replacement')
        [plt.axvline(x=_inspec_times, color='grey', alpha=0.2, linestyle='--') for _inspec_times in inspec_times]
        # TODO: gérer la légende
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("$R(t)$")
        plt.title(f'$k(l) = {kl}$, $k(l_0) = {kl0}$. Lifetime = ${round(lifetime, 3)}$')
        plt.show()

    def empirical_one_cycle_cost(self, strategy, cost_structure, nb_sample):

        control_frequency, replacement_threshold = strategy
        cI, cP, cF = cost_structure
        T_max = 1
        while self.sf(T_max) > 1e-4:
            T_max = T_max + 1
        inspection_times = np.linspace(0, T_max, num=1000)
        data = self.sample(inspection_times=inspection_times, nb_sample=nb_sample)

        empirical_cost = []
        control_times = np.arange(1, 100) * control_frequency

        for i in range(nb_sample):

            deterioration_measurements = self.r0 - data.deterioration_measurements[data.ids == i]
            kl = self.count_greater(replacement_threshold, inspection_times, deterioration_measurements, control_times)
            kl0 = self.count_greater(self.l0, inspection_times, deterioration_measurements, control_times)
            epsilon_l_l0 = kl0 - kl
            ind_defaillance = np.where(deterioration_measurements < self.l0)[0][0]
            t_defaillance = inspection_times[ind_defaillance]
            if epsilon_l_l0 > 0:
                lifetime = control_times[kl]
                empirical_cost.append((kl * cI + cP) / lifetime)
            else:
                lifetime = t_defaillance
                empirical_cost.append((kl * cI + cF) / lifetime)
        return np.mean(empirical_cost)

    def theoretical_one_cycle_cost(self, strategy, cost_structure, tol=1e-4, print_nit=False):

        control_frequency, replacement_threshold = strategy
        cI, cP, cF = cost_structure

        res = []
        accuracy = np.inf

        i = 0
        while accuracy > tol:
            if i == 0:
                res.append(

                    (i * cI + cP) / ((i + 1) * control_frequency) * (gamma.cdf(self.r0 - self.l0, a=self.shape_function(
                        (i + 1) * control_frequency) - self.shape_function(i * control_frequency),
                                                                               scale=1 / self.rate) - gamma.cdf(
                        self.r0 - replacement_threshold,
                        a=self.shape_function((i + 1) * control_frequency) - self.shape_function(
                            i * control_frequency),
                        scale=1 / self.rate))

                    + (i * cI + cF) * (integrate.quad(
                        func=lambda t: t ** (-2) * (
                                1 - gamma.cdf(self.r0 - self.l0,
                                              a=self.shape_function(t) - self.shape_function(i * control_frequency),
                                              scale=1 / self.rate)),
                        a=0,
                        b=control_frequency)[0]

                                       + control_frequency ** (-1) * (
                                               1 - gamma.cdf(self.r0 - self.l0,
                                                             a=self.shape_function(
                                                                 (i + 1) * control_frequency) - self.shape_function(
                                                                 i * control_frequency),
                                                             scale=1 / self.rate)))
                )
            else:
                res.append(

                    (i * cI + cP) / ((i + 1) * control_frequency) * integrate.quad(
                        func=lambda x: (gamma.cdf(self.r0 - self.l0 - x,
                                                  a=self.shape_function(
                                                      (i + 1) * control_frequency) - self.shape_function(
                                                      i * control_frequency),
                                                  scale=1 / self.rate)
                                        - gamma.cdf(self.r0 - replacement_threshold - x,
                                                    a=self.shape_function(
                                                        (i + 1) * control_frequency) - self.shape_function(
                                                        i * control_frequency),
                                                    scale=1 / self.rate)) * gamma.pdf(x, a=self.shape_function(
                            i * control_frequency), scale=1 / self.rate), a=0, b=self.r0 - replacement_threshold)[0]

                    + (i * cI + cF) * (integrate.dblquad(
                        func=lambda t, x: t ** (-2) * (1 - gamma.cdf(self.r0 - self.l0 - x,
                                                                     a=self.shape_function(t) - self.shape_function(
                                                                         i * control_frequency),
                                                                     scale=1 / self.rate)) *
                                          gamma.pdf(x, a=self.shape_function(i * control_frequency),
                                                    scale=1 / self.rate),
                        gfun=i * control_frequency,
                        hfun=(i + 1) * control_frequency,
                        a=0,
                        b=self.r0 - replacement_threshold)[0] + ((i + 1) * control_frequency) ** (-1)
                                       * integrate.quad(
                                func=lambda x: (1 - gamma.cdf(self.r0 - self.l0 - x,
                                                              a=self.shape_function((i + 1) * control_frequency)
                                                                - self.shape_function(i * control_frequency),
                                                              scale=1 / self.rate))
                                               * gamma.pdf(x,
                                                           a=self.shape_function(i * control_frequency),
                                                           scale=1 / self.rate),
                                a=0,
                                b=self.r0 - replacement_threshold)[0])

                )
                if np.sum(res[:-1]) == 0:
                    accuracy = np.inf
                else:
                    accuracy = res[-1] / np.sum(res[:-1])
            i += 1
        if print_nit:
            print(f"{i} iterations were made before convergence")
        return np.sum(res)

    def theoretical_one_cycle_cost_minimizer(self, cost_structure):
        q80 = 0
        while self.sf(q80) > 0.2:
            q80 += 1
        tau_initial_guess = q80
        l_initial_guess = (self.r0 + self.l0) / 2
        opt = minimize(
            fun=self.theoretical_one_cycle_cost,
            x0=[tau_initial_guess, l_initial_guess],
            args=cost_structure,
            bounds=((1, np.inf), (self.l0, self.r0)),
            method='SLSQP'
        )
        return opt.x
