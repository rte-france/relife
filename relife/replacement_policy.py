"""Run-to-failure and age replacement maintenance policies."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

from __future__ import annotations
import numpy as np
import scipy.optimize as optimize
from typing import Tuple

from .model import AbsolutelyContinuousLifetimeModel, AgeReplacementModel, LeftTruncated
from .data import ReplacementPolicyData
from .reward import FailureCost, AgeReplacementCost
from .discounting import ExponentialDiscounting
from .renewal_process import RenewalRewardProcess
from .utils import args_size, args_take, args_ndim, gauss_legendre


# One Cyle Replacement Policies


class OneCycleRunToFailure:
    """One cyle run-to-failure policy."""

    reward: FailureCost = FailureCost()  #: The failure cost of the asset.
    discount: ExponentialDiscounting = (
        ExponentialDiscounting()
    )  #: Exponential discounting.

    def __init__(
        self,
        model: AbsolutelyContinuousLifetimeModel,
        args:Tuple[np.ndarray,...] = (),
        a0: np.ndarray = None,
        cf: np.ndarray = None,
        rate: np.ndarray = 0,
    ) -> None:
        """One cycle run-to-failure policy.

        Parameters
        ----------
        model : AbsolutelyContinuousLifetimeModel
            Absolutely continuous lifetime model of the asset.
        args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model.
        a0 : float or 2D array, optional
            Current ages of the assets, by default 0 for each asset.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default 0.

        Notes
        -----
        If `cf` is set to None, if should be defined when using methods to
        compute costs.

        If `cf` and `rate` are 2D or 3D array, then:

        - axis=-2 represents the indices of each asset,
        - axis=-3 represents the indices of each component of the cost vector.

        References
        ----------
        .. [1] Coolen-Schrijner, P., & Coolen, F. P. A. (2006). On optimality
            criteria for age replacement. Proceedings of the Institution of
            Mechanical Engineers, Part O: Journal of Risk and Reliability,
            220(1), 21-29
        """
        if a0 is not None:
            model = LeftTruncated(model)
            args = (a0, *args)
        self.model = model
        self.args = args
        self.cf = cf
        self.rate = rate

    def _parse_policy_args(
        self, cf: np.ndarray, rate: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Parse the arguments of the policy.

        Parameters
        ----------
        cf : float, 2D array or 3D array
            Costs of failures.
        rate : float, 2D array or 3D array
            Discount rate.

        Returns
        -------
        Tuple[ndarray,...]
            `(cf, rate)`

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        if cf is None:
            cf = self.cf
        if rate is None:
            rate = self.rate
        return cf, rate

    def rrp_args(
        self, cf: np.ndarray = None, rate: np.ndarray = None
    ) -> Tuple[Tuple[np.ndarray, ...], ...]:
        """Arguments of the underlying renewal reward process.

        Parameters
        ----------
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        Tuple[Tuple[ndarray,...],...]
            `(model_args, reaward_args, discount_args)`

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        cf, rate = self._parse_policy_args(cf, rate)
        model_args = self.args
        reward_args = (cf,)
        discount_args = (rate,)
        return model_args, reward_args, discount_args

    def expected_total_cost(
        self, t: np.ndarray, cf: np.ndarray = None, rate: np.ndarray = None
    ) -> np.ndarray:
        """The expected total discounted cost.

        Parameters
        ----------
        t : 1D array
            Timeline.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        ndarray
            The cumulative expected total cost for each asset along the
            timeline.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        return RenewalRewardProcess._reward_partial_expectation(
            t, self.model, self.reward, self.discount, *self.rrp_args(cf, rate)
        )

    def asymptotic_expected_total_cost(
        self, cf: np.ndarray = None, rate: np.ndarray = None
    ) -> np.ndarray:
        """The asymptotic expected total cost.

        Parameters
        ----------
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        ndarray
            The asymptotic expected total cost for each asset.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        return RenewalRewardProcess._reward_partial_expectation(
            np.inf, self.model, self.reward, self.discount, *self.rrp_args(cf, rate)
        )

    def expected_equivalent_annual_cost(
        self,
        t: np.ndarray,
        cf: np.ndarray = None,
        rate: np.ndarray = None,
        dt: float = 1.0,
    ) -> np.ndarray:
        r"""The expected equivalent annual cost.

        Parameters
        ----------
        t : 1D array
            Timeline.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.
        dt : float, optional
            The length of the first period before discounting, by default 1.

        Returns
        -------
        ndarray
            The expected equivalent annual cost until time `t`.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.

        The expected equivalent annual cost until time :math:`t` is given by:

        .. math::

            EEAC(t) = \int_0^t \frac{\delta c_f e^{-\delta x}}{1 - e^{-\delta
            x}} \mathrm{d}F(x)
        """
        model_args, reward_args, discount_args = self.rrp_args(cf, rate)
        ndim = args_ndim(t, *model_args, *reward_args, *discount_args)
        f = (
            lambda x: self.reward.conditional_expectation(x, *reward_args)
            * self.discount.factor(x, *discount_args)
            / self.discount.annuity_factor(x, *discount_args)
        )
        mask = t < dt
        q0 = self.model.cdf(dt, *model_args) * f(dt)
        return q0 + np.where(
            mask, 0, self.model.ls_integrate(f, dt, t, *model_args, ndim=ndim)
        )

    def asymptotic_expected_equivalent_annual_cost(
        self, cf: np.ndarray = None, rate: np.ndarray = None, dt: float = 1.0
    ) -> np.ndarray:
        r"""The asymptotic expected equivalent annual cost.

        Parameters
        ----------
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.
        dt : float, optional
            The length of the first period before discounting, by default 1.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.

        The asymptotic expected equivalent annual cost is:

        .. math::

            EEAC_\infty = \int_0^\infty \frac{\delta c_f e^{-\delta x}}{1 -
            e^{-\delta x}} \mathrm{d}F(x)
        """
        return self.expected_equivalent_annual_cost(np.inf, cf, rate, dt)

    def sample(
        self,
        cf: np.ndarray = None,
        rate: np.ndarray = None,
        n_samples: int = 1,
        random_state: int = None,
    ) -> ReplacementPolicyData:
        """One cycle run-to-failure policy sampling.

        Parameters
        ----------
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.
        n_samples : int, optional
            Number of samples, by default 1.
        random_state : int, optional
            Random seed, by default None.

        Returns
        -------
        ReplacementPolicyData
            Samples of replacement times, durations, costs and events for each asset.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        T = np.inf
        model_args, reward_args, discount_args = self.rrp_args(cf, rate)
        n_indices = max(1, args_size(*model_args, *reward_args))
        data = RenewalRewardProcess._sample_init(
            T,
            self.model,
            self.reward,
            self.discount,
            model_args,
            reward_args,
            discount_args,
            n_indices,
            n_samples,
            random_state,
        )
        events = np.ones(data.size)
        if isinstance(self.model, LeftTruncated):
            a0, *args = args_take(data.indices, *self.args)
        else:
            args = args_take(data.indices, *self.args)
            a0 = np.zeros(data.indices.size).reshape(-1, 1)
        return ReplacementPolicyData(*data.astuple(), events, args, a0)


class OneCycleAgeReplacementPolicy:
    """One-cyle age replacement policy.

    The asset is disposed at a fixed age `ar` with costs `cp` or upon failure
    with costs `cf` if earlier.
    """

    reward: AgeReplacementCost = AgeReplacementCost()  #: Costs of the replacement.
    discount: ExponentialDiscounting = (
        ExponentialDiscounting()
    )  #: Exponential discounting.

    def __init__(
        self,
        model: AbsolutelyContinuousLifetimeModel,
        args:Tuple[np.ndarray,...]=(),
        a0: np.ndarray = None,
        ar: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = 0,
    ) -> None:
        """One-cyle age replacement policy.

        Parameters
        ----------
        model : AbsolutelyContinuousLifetimeModel
            Absolutely continuous lifetime model of the asset.
        args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model, by default ().
        a0 : float or 2D array, optional
            Current ages of the assets, by default 0 for each asset.
        ar : float, 2D array, optional
            Ages of preventive replacements, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default 0.

        Notes
        -----
        If `ar`, `cf` or `cp` is set to None, the argument should be defined
        when using methods to compute costs.

        If `cf`, `cp` and `rate` are 2D or 3D array, then:

        - axis=-2 represents the indices of each asset,
        - axis=-3 represents the indices of each component of the cost vector.

        References
        ----------
        .. [1] Coolen-Schrijner, P., & Coolen, F. P. A. (2006). On optimality
            criteria for age replacement. Proceedings of the Institution of
            Mechanical Engineers, Part O: Journal of Risk and Reliability,
            220(1), 21-29
        """
        if a0 is not None:
            model = LeftTruncated(model)
            args = (a0, *args)
        self.model = AgeReplacementModel(model)
        self.args = args
        self.ar = ar
        self.cf = cf
        self.cp = cp
        self.rate = rate

    def _parse_policy_args(
        self, ar: np.ndarray, cf: np.ndarray, cp: np.ndarray, rate: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Parse the arguments of the policy.

        Parameters
        ----------
        ar : float or 2D array
            Ages of preventive replacements.
        cf : float, 2D array or 3D array
            Costs of failures.
        cp : float, 2D array or 3D array
            Costs of preventive replacements.
        rate : float, 2D array or 3D array
            Discount rate.

        Returns
        -------
        Tuple[ndarray,...]
            `(ar, cf, cp, rate)`

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        if ar is None:
            ar = self.ar
        if cf is None:
            cf = self.cf
        if cp is None:
            cp = self.cp
        if rate is None:
            rate = self.rate
        return ar, cf, cp, rate

    def rrp_args(
        self,
        ar: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
    ) -> Tuple[Tuple[np.ndarray, ...], ...]:
        """Arguments of the underlying renewal reward process.

        Parameters
        ----------
        ar : float or 2D array, optional
            Ages of preventive replacements, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        Tuple[Tuple[ndarray,...],...]
            `(model_args, reaward_args, discount_args)`

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        ar, cf, cp, rate = self._parse_policy_args(ar, cf, cp, rate)
        model_args = (ar, *self.args)
        reward_args = (ar, cf, cp)
        discount_args = (rate,)
        return model_args, reward_args, discount_args

    @classmethod
    def optimal_replacement_age(
        cls,
        model: AbsolutelyContinuousLifetimeModel,
        cf: np.ndarray,
        cp: np.ndarray,
        rate: np.ndarray = 0,
        args:Tuple[np.ndarray,...]=(),
    ) -> np.ndarray:
        r"""Compute the optimal age of preventive replacement for each asset.

        The optimal age of preventive replacement is computed by minimizing the
        asymptotic expected equivalent annual cost on one-cycle.

        Parameters
        ----------
        model : AbsolutelyContinuousLifetimeModel
            Absolutely continuous lifetime model.
        cf : float, 2D array or 3D array
            Costs of failures.
        cp : float, 2D array or 3D array
            Costs of preventive replacements.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default 0.
        args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model, by default ().

        Returns
        -------
        ndarray
            The optimal age of preventive replacement.

        Notes
        -----
        The optimal age of replacement minimizes the asymptotic expected
        equivalent annual cost:

        .. math::

            EEAC_\infty(a) = \sum_i {c_f}_i \int_0^a \dfrac{\delta_i e^{-\delta_i
            t}}{1 - e^{-\delta_i t}} f(t) \mathrm{d}t + {c_p}_i S(a)\dfrac{\delta_i
            e^{-\delta_i a}}{1 - e^{-\delta_i a}}

        where:

        - :math:`a` is the age of replacement,
        - :math:`{c_f}_i, {c_p}_i, \delta_i` are respectively the components of
          the failures costs, preventive costs and the associated discount rate,
        - :math:`S, f, h` are respectively the survival function, the
          probability density function and the hazard function of the underlying
          lifetime model.

        The optimal age of replacement is then solution of the equation:
        
        .. math::

            \sum_i \dfrac{\delta_i e^{-\delta_i a}}{1 - e^{-\delta_i a}} \left(
            ({c_f}_i - {c_p}_i) h(a) - \dfrac{\delta_i {c_p}_i}{1 - e^{-\delta_i
            a}} \right) = 0
        """
        size = args_size(cf, cp, *args)
        cf, cp = np.array(cf, ndmin=3), np.array(cp, ndmin=3)
        x0 = np.minimum(np.sum(cp, axis=0) / np.sum(cf - cp, axis=0), 1)
        if np.size(x0) == 1:
            x0 = np.tile(x0, (size, 1))
        eq = lambda a: np.sum(
            cls.discount.factor(a, rate)
            / cls.discount.annuity_factor(a, rate)
            * (
                (cf - cp) * model.hf(a, *args)
                - cp / cls.discount.annuity_factor(a, rate)
            ),
            axis=0,
        )
        ar = optimize.newton(eq, x0)
        return ar.squeeze() if np.size(ar) == 1 else ar

    def fit(
        self, cf: np.ndarray = None, cp: np.ndarray = None, rate: np.ndarray = None
    ) -> OneCycleAgeReplacementPolicy:
        """Computes and sets the optimal age of replacement for each asset.

        Parameters
        ----------
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        self
            The fitted policy as the current object.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        _, cf, cp, rate = self._parse_policy_args(None, cf, cp, rate)
        self.ar = self.optimal_replacement_age(
            self.model.baseline, cf, cp, rate, self.args
        )
        return self

    def expected_total_cost(
        self,
        t: np.ndarray,
        ar: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
    ) -> np.ndarray:
        """The expected total discounted cost.

        Parameters
        ----------
        t : 1D array
            Timeline.
        ar : float or 2D array, optional
            Ages of replacement, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        ndarray
            The cumulative expected total cost for each asset along the
            timeline.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        return RenewalRewardProcess._reward_partial_expectation(
            t, self.model, self.reward, self.discount, *self.rrp_args(ar, cf, cp, rate)
        )

    def asymptotic_expected_total_cost(
        self,
        ar: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
    ) -> np.ndarray:
        """The asymptotic expected total cost.

        Parameters
        ----------
        ar : float or 2D array, optional
            Ages of preventive replacement, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        ndarray
            The asymptotic expected total cost for each asset.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        return RenewalRewardProcess._reward_partial_expectation(
            np.inf,
            self.model,
            self.reward,
            self.discount,
            *self.rrp_args(ar, cf, cp, rate),
        )

    def expected_equivalent_annual_cost(
        self,
        t: np.ndarray,
        ar: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
        dt: float = 1.0,
    ) -> np.ndarray:
        r"""The expected equivalent annual cost.

        Parameters
        ----------
        t : 1D array
            Timeline.
        ar : float or 2D array, optional
            Ages of preventive replacement, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.
        dt : float, optional
            The length of the first period before discounting, by default 1.

        Returns
        -------
        ndarray
            The cumulative expected equivalent annual cost until time t.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.

        The expected equivalent annual cost until time :math:`t` is:

        .. math::

            EEAC(t) = \int_0^t \frac{\delta c_f e^{-\delta x}}{1 - e^{-\delta
            x}} \mathrm{d}F(x)
        """
        model_args, reward_args, discount_args = self.rrp_args(ar, cf, cp, rate)
        ndim = args_ndim(t, *model_args, *reward_args, *discount_args)
        f = (
            lambda x: self.reward.conditional_expectation(x, *reward_args)
            * self.discount.factor(x, *discount_args)
            / self.discount.annuity_factor(x, *discount_args)
        )
        mask = t < dt
        q0 = self.model.cdf(dt, *model_args) * f(dt)
        return q0 + np.where(
            mask, 0, self.model.ls_integrate(f, dt, t, *model_args, ndim=ndim)
        )

    def asymptotic_expected_equivalent_annual_cost(
        self,
        ar: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
        dt: float = 1.0,
    ) -> np.ndarray:
        r"""The asymptotic expected equivalent annual cost.

        Parameters
        ----------
        ar : float or 2D array, optional
            Ages of preventive replacement, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.
        dt : float, optional
            The length of the first period before discounting, by default 1.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.

        The asymptotic expected equivalent annual cost is:

        .. math::

            EEAC_\infty = \int_0^\infty \frac{\delta c_f e^{-\delta x}}{1 -
            e^{-\delta x}} \mathrm{d}F(x)
        """
        return self.expected_equivalent_annual_cost(np.inf, ar, cf, cp, rate, dt)

    def sample(
        self,
        ar: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
        n_samples: int = 1,
        random_state: int = None,
    ) -> ReplacementPolicyData:
        """One-cycle age replacement policy sampling.

        Parameters
        ----------
        ar : float or 2D array, optional
            Ages of preventive replacement, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.
        n_samples : int, optional
            Number of samples, by default 1.
        random_state : int, optional
            Random seed, by default None.

        Returns
        -------
        ReplacementPolicyData
            Samples of replacement times, durations, costs and events for each asset.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        T = np.inf
        model_args, reward_args, discount_args = self.rrp_args(ar, cf, cp, rate)
        n_indices = max(1, args_size(*model_args, *reward_args))
        data = RenewalRewardProcess._sample_init(
            T,
            self.model,
            self.reward,
            self.discount,
            model_args,
            reward_args,
            discount_args,
            n_indices,
            n_samples,
            random_state,
        )
        ar = model_args[0]
        events = np.where(data.durations < args_take(data.indices, ar)[0].ravel(), 1, 0)
        if isinstance(self.model.baseline, LeftTruncated):
            a0, *args = args_take(data.indices, *self.args)
        else:
            args = args_take(data.indices, *self.args)
            a0 = np.zeros(data.indices.size).reshape(-1, 1)
        return ReplacementPolicyData(*data.astuple(), events, args, a0)


# Renewal Process Replacement Policy


class RunToFailure:
    """Run-to-failure renewal policy."""

    reward: FailureCost = FailureCost()  #: The failure cost of the asset.
    discount: ExponentialDiscounting = (
        ExponentialDiscounting()
    )  #: Exponential discounting.

    def __init__(
        self,
        model: AbsolutelyContinuousLifetimeModel,
        args:Tuple[np.ndarray,...]=(),
        a0: np.ndarray = None,
        cf: np.ndarray = None,
        rate: np.ndarray = 0,
    ) -> None:
        """Run-to-failure renwal policy.

        Renewal reward process where assets are replaced on failure with costs
        `cf`.

        Parameters
        ----------
        model : AbsolutelyContinuousLifetimeModel
            Absolutely continuous lifetime model of the asset.
        args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model, by default ().
        a0 : float or 2D array, optional
            Current ages of the assets, by default 0 for each asset.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default 0.

        Notes
        -----
        If `cf` is set to None, if should be defined when using methods to
        compute costs.

        If `cf` and `rate` are 2D or 3D array, then:

        - axis=-2 represents the indices of each asset,
        - axis=-3 represents the indices of each component of the cost vector.

        References
        ----------
        .. [1] Van der Weide, J. A. M., & Van Noortwijk, J. M. (2008). Renewal
            theory with exponential and hyperbolic discounting. Probability in
            the Engineering and Informational Sciences, 22(1), 53-74.
        """
        if a0 is not None:
            self.model1 = LeftTruncated(model)
            self.args1 = (a0, *args)
        else:
            self.model1 = None
            self.args1 = ()
        self.model = model
        self.args = args
        self.cf = cf
        self.rate = rate
        self.rrp = RenewalRewardProcess(
            self.model, self.reward, self.model1, self.reward, self.discount
        )

    def _parse_policy_args(
        self, cf: np.ndarray, rate: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Parse the arguments of the policy.

        Parameters
        ----------
        cf : float, 2D array or 3D array
            Costs of failures.
        rate : float, 2D array or 3D array
            Discount rate.

        Returns
        -------
        Tuple[ndarray,...]
            `(cf, rate)`

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        if cf is None:
            cf = self.cf
        if rate is None:
            rate = self.rate
        return cf, rate

    def rrp_args(
        self, cf: np.ndarray = None, rate: np.ndarray = None
    ) -> Tuple[Tuple[np.ndarray, ...], ...]:
        """Arguments of the underlying renewal reward process.

        Parameters
        ----------
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        Tuple[Tuple[ndarray,...],...]
            `(model_args, reward_args, model1_args, reward1_args, discount_args)`

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        cf, rate = self._parse_policy_args(cf, rate)
        model_args = self.args
        reward_args = (cf,)
        model1_args = self.args1
        reward1_args = (cf,)
        discount_args = (rate,)
        return model_args, reward_args, model1_args, reward1_args, discount_args

    def expected_total_cost(
        self, t: np.ndarray, cf: np.ndarray = None, rate: np.ndarray = None
    ) -> np.ndarray:
        """The expected total discounted cost.

        The expected total discounted cost is computed by solving the renewal
        equation.

        Parameters
        ----------
        t : 1D array
            Timeline.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        ndarray
            The cumulative expected total cost for each asset along the
            timeline.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        return self.rrp.expected_total_reward(t, *self.rrp_args(cf, rate))

    def asymptotic_expected_total_cost(
        self, cf: np.ndarray = None, rate: np.ndarray = None
    ) -> np.ndarray:
        """The asymptotic expected total cost.

        Parameters
        ----------
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        ndarray
            The asymptotic expected total cost for each asset.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        return self.rrp.asymptotic_expected_total_reward(*self.rrp_args(cf, rate))

    def expected_equivalent_annual_cost(
        self, t: np.ndarray, cf: np.ndarray = None, rate: np.ndarray = None
    ) -> np.ndarray:
        r"""The expected equivalent annual cost.

        where :math:`z` is the expected total cost.

        Parameters
        ----------
        t : 1D array
            Timeline.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        ndarray
            The expected equivalent annual cost for each asset along the
            timeline.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.

        The expected equivalent annual cost at time :math:`t` is:

        .. math::

            EEAC(t) = \dfrac{\delta z(t)}{1 - e^{-\delta t}}
        """
        return self.rrp.expected_equivalent_annual_worth(t, *self.rrp_args(cf, rate))

    def asymptotic_expected_equivalent_annual_cost(
        self, cf: np.ndarray = None, rate: np.ndarray = None
    ) -> np.ndarray:
        r"""The asymptotic expected equivalent annual cost.

        Parameters
        ----------
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost for each asset.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.

        The asymptotic expected equivalent annual cost is:
        
        .. math::

            EEAC_\infty = \lim_{t \to \infty} EEAC(t)
        """
        return self.rrp.asymptotic_expected_equivalent_annual_worth(
            *self.rrp_args(cf, rate)
        )

    def sample(
        self,
        T: float,
        cf: np.ndarray = None,
        rate: np.ndarray = None,
        n_samples: int = 1,
        random_state: int = None,
    ) -> ReplacementPolicyData:
        """Run-to-failure renewal policy sampling.

        Parameters
        ----------
        T : float
            End of the observation period.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.
        n_samples : int, optional
            Number of samples, by default 1.
        random_state : int, optional
            Random seed, by default None.

        Returns
        -------
        ReplacementPolicyData
            Samples of replacement times, durations, costs and events for each asset.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        data = self.rrp.sample(T, *self.rrp_args(cf, rate), n_samples, random_state)
        events = np.ones(data.size)
        args = args_take(data.indices, *self.args)
        a0 = np.zeros(data.size)
        if self.model1 is not None:
            ind1 = data.n_indices * data.n_samples
            a0[:ind1] = args_take(data.indices[:ind1], *self.args1)[0]
        return ReplacementPolicyData(*data.astuple(), events, args, a0)


class AgeReplacementPolicy:
    """Time based replacement policy."""

    reward: AgeReplacementCost = AgeReplacementCost()  #: Costs of the replacement.
    discount: ExponentialDiscounting = (
        ExponentialDiscounting()
    )  #: Exponential discounting.

    def __init__(
        self,
        model: AbsolutelyContinuousLifetimeModel,
        args:Tuple[np.ndarray,...]=(),
        a0: np.ndarray = None,
        ar: np.ndarray = None,
        ar1: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = 0,
    ) -> None:
        """Age replacement renewal policy.

        Renewal reward process where assets are replaced at a fixed age `ar`
        with costs `cp` or upon failure with costs `cf` if earlier.

        Parameters
        ----------
        model : AbsolutelyContinuousLifetimeModel
            Absolutely continuous lifetime model of the asset.
        args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model, by default ().
        a0 : float or 2D array, optional
            Current ages of the assets, by default 0 for each asset.
        ar : float, 2D array, optional
            Ages of preventive replacements, by default None.
        ar1 : float, 2D array, optional
            Ages of the first preventive replacements, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default 0.

        Notes
        -----
        If `ar`, `ar1`, `cf` or `cp` is set to None, the argument should be
        defined when using methods to compute costs.

        If `cf`, `cp` and `rate` are 2D or 3D array, then:

        - axis=-2 represents the indices of each asset,
        - axis=-3 represents the indices of each component of the cost vector.

        References
        ----------
        .. [1] Mazzuchi, T. A., Van Noortwijk, J. M., & Kallen, M. J. (2007).
            Maintenance optimization. Encyclopedia of Statistics in Quality and
            Reliability, 1000-1008.
        """
        if a0 is not None:
            self.model1 = AgeReplacementModel(LeftTruncated(model))
            self.args1 = (a0, *args)
        else:
            self.model1 = None
            self.args1 = ()
        self.model = AgeReplacementModel(model)
        self.args = args
        self.ar = ar
        self.ar1 = ar1
        self.cf = cf
        self.cp = cp
        self.rate = rate
        self.rrp = RenewalRewardProcess(
            self.model, self.reward, self.model1, self.reward, self.discount
        )

    def _parse_policy_args(
        self,
        ar: np.ndarray,
        ar1: np.ndarray,
        cf: np.ndarray,
        cp: np.ndarray,
        rate: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """Parse the arguments of the policy.

        Parameters
        ----------
        ar : float or 2D array
            Ages of preventive replacements.
        ar1 : float, 2D array
            Ages of the first preventive replacements.
        cf : float, 2D array or 3D array
            Costs of failures.
        cp : float, 2D array or 3D array
            Costs of preventive replacements.
        rate : float, 2D array or 3D array
            Discount rate.

        Returns
        -------
        Tuple[ndarray,...]
            `(ar, ar1, cf, cp, rate)`

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        if ar is None:
            ar = self.ar
        if ar1 is None:
            ar1 = self.ar1
        if cf is None:
            cf = self.cf
        if cp is None:
            cp = self.cp
        if rate is None:
            rate = self.rate
        return ar, ar1, cf, cp, rate

    def rrp_args(
        self,
        ar: np.ndarray = None,
        ar1: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
    ) -> Tuple[Tuple[np.ndarray, ...], ...]:
        """Arguments of the underlying renewal reward process.

        Parameters
        ----------
        ar : float, 2D array, optional
            Ages of preventive replacements, by default None.
        ar1 : float, 2D array, optional
            Ages of the first preventive replacements, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default 0.

        Returns
        -------
        Tuple[Tuple[ndarray,...],...]
            `(model_args, reward_args, model1_args, reward1_args, discount_args)`

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        ar, ar1, cf, cp, rate = self._parse_policy_args(ar, ar1, cf, cp, rate)
        model_args = (ar, *self.args)
        reward_args = (ar, cf, cp)
        model1_args = (ar1, *self.args1)
        reward1_args = (ar1, cf, cp)
        discount_args = (rate,)
        return model_args, reward_args, model1_args, reward1_args, discount_args

    @classmethod
    def optimal_replacement_age(
        cls,
        model: AbsolutelyContinuousLifetimeModel,
        cf: np.ndarray,
        cp: np.ndarray,
        rate: np.ndarray = 0,
        args:Tuple[np.ndarray,...]=(),
    ) -> np.ndarray:
        r"""Compute the optimal age of preventive replacement for each asset.

        The optimal age of preventive replacement is computed by minimizing the
        asymptotic expected equivalent annual cost of the renewal reward
        process.

        Parameters
        ----------
        model : AbsolutelyContinuousLifetimeModel
            Absolutely continuous lifetime model.
        cf : float, 2D array or 3D array
            Costs of failures.
        cp : float, 2D array or 3D array
            Costs of preventive replacements.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default 0.
        args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model, by default ().

        Returns
        -------
        ndarray
            The optimal age of preventive replacement.

        Notes
        -----
        The optimal age of replacement minimizes the asymptotic expected
        equivalent annual cost:

        .. math::

            EEAC_\infty(a) = \sum_i \dfrac{{c_f}_i \int_0^a e^{-\delta_i x} f(x)
            \mathrm{d}x + {c_p}_i e^{-\delta_i a} S(a)}{\int_0^a e^{-\delta_i x}
            S(x) \mathrm{d}x} 

        where:

        - :math:`a` is the age of replacement,
        - :math:`{c_f}_i, {c_p}_i, \delta_i` are respectively the components of
          the failures costs, preventive costs and the associated discount rate,
        - :math:`S, f, h` are respectively the survival function, the
          probability density function and the hazard function of the underlying
          lifetime model.

        The optimal age of replacement is then solution of the equation:
        
        .. math::

            \sum_i \dfrac{\left( ({c_f}_i - {c_p}_i) \left(
                h(a) \int_0^a e^{-\delta_i x} S(x) \mathrm{d}x - \int_0^a
                e^{-\delta_i x} f(x) \mathrm{d}x  \right) - {c_p}_i \right)}
                {\left( \int_0^a e^{-\delta_i x} S(x) \mathrm{d}x \right)^2} = 0
        """
        size = args_size(cf, cp, *args)
        cf, cp = np.array(cf, ndmin=3), np.array(cp, ndmin=3)
        ndim = args_ndim(cf, cp, rate, *args)
        x0 = np.minimum(np.sum(cp, axis=0) / np.sum(cf - cp, axis=0), 1)
        if np.size(x0) == 1:
            x0 = np.tile(x0, (size, 1))
        f = lambda x: cls.discount.factor(x, rate) * model.sf(x, *args)
        g = lambda x: cls.discount.factor(x, rate) * model.pdf(x, *args)

        def eq(a):
            F = gauss_legendre(f, 0, a, ndim=ndim)
            G = gauss_legendre(g, 0, a, ndim=ndim)
            return np.sum(
                cls.discount.factor(a, rate)
                * ((cf - cp) * (model.hf(a, *args) * F - G) - cp)
                / F**2,
                axis=0,
            )

        ar = optimize.newton(eq, x0)
        return ar.squeeze() if np.size(ar) == 1 else ar

    def fit(
        self,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
        fit_ar1: bool = True,
    ) -> AgeReplacementPolicy:
        """Computes and sets the optimal age of replacement for each asset.

        Parameters
        ----------
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.
        fit_ar1 : bool, optional
            If True, computes and sets the optimal age of replacement for the
            first replacement considering a one-cycle age replacement criteria.

        Returns
        -------
        self
            The fitted policy as the current object.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        _, _, cf, cp, rate = self._parse_policy_args(None, None, cf, cp, rate)
        self.ar = self.optimal_replacement_age(
            self.model.baseline, cf, cp, rate, self.args
        )
        if self.model1 is not None and fit_ar1:
            self.ar1 = OneCycleAgeReplacementPolicy.optimal_replacement_age(
                self.model1.baseline, cf, cp, rate, self.args1
            )
        return self

    def expected_total_cost(
        self,
        t: np.ndarray,
        ar: np.ndarray = None,
        ar1: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
    ) -> np.ndarray:
        """The expected total discounted cost.

        The expected total discounted cost is computed by solving the renewal
        equation.

        Parameters
        ----------
        t : 1D array
            Timeline.
        ar : float or 2D array, optional
            Ages of replacement, by default None.
        ar1 : float, 2D array, optional
            Ages of the first preventive replacements, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        ndarray
            The cumulative expected total cost for each asset along the
            timeline.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        return self.rrp.expected_total_reward(t, *self.rrp_args(ar, ar1, cf, cp, rate))

    def asymptotic_expected_total_cost(
        self,
        ar: np.ndarray = None,
        ar1: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
    ) -> np.ndarray:
        """The asymptotic expected total cost.

        Parameters
        ----------
        ar : float or 2D array, optional
            Ages of preventive replacement, by default None.
        ar1 : float, 2D array, optional
            Ages of the first preventive replacements, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        ndarray
            The asymptotic expected total cost for each asset.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        return self.rrp.asymptotic_expected_total_reward(
            *self.rrp_args(ar, ar1, cf, cp, rate)
        )

    def expected_equivalent_annual_cost(
        self,
        t: np.ndarray,
        ar: np.ndarray = None,
        ar1: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
    ) -> np.ndarray:
        r"""The expected equivalent annual cost.

        Parameters
        ----------
        t : 1D array
            Timeline.
        ar : float or 2D array, optional
            Ages of replacement, by default None.
        ar1 : float, 2D array, optional
            Ages of the first preventive replacements, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        ndarray
            The expected equivalent annual cost for each asset along the
            timeline.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.

        The expected equivalent annual cost at time :math:`t` is:

        .. math::

            EEAC(t) = \dfrac{\delta z(t)}{1 - e^{-\delta t}}

        where :math:`z` is the expected total cost.
        """
        return self.rrp.expected_equivalent_annual_worth(
            t, *self.rrp_args(ar, ar1, cf, cp, rate)
        )

    def asymptotic_expected_equivalent_annual_cost(
        self,
        ar: np.ndarray = None,
        ar1: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
    ) -> np.ndarray:
        r"""The asymptotic expected equivalent annual cost.

        Parameters
        ----------
        ar : float or 2D array, optional
            Ages of replacement, by default None.
        ar1 : float, 2D array, optional
            Ages of the first preventive replacements, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost for each asset.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.

        The asymptotic expected equivalent annual cost is:
        
        .. math::

            EEAC_\infty = \lim_{t \to \infty} EEAC(t)
        """
        return self.rrp.asymptotic_expected_equivalent_annual_worth(
            *self.rrp_args(ar, ar1, cf, cp, rate)
        )

    def sample(
        self,
        T: float,
        ar: np.ndarray = None,
        ar1: np.ndarray = None,
        cf: np.ndarray = None,
        cp: np.ndarray = None,
        rate: np.ndarray = None,
        n_samples: int = 1,
        random_state: int = None,
    ) -> ReplacementPolicyData:
        """Age replacement renewal policy sampling.

        Parameters
        ----------
        T : float
            End of the observation period.
        ar : float or 2D array, optional
            Ages of replacement, by default None.
        ar1 : float, 2D array, optional
            Ages of the first preventive replacements, by default None.
        cf : float, 2D array or 3D array, optional
            Costs of failures, by default None.
        cp : float, 2D array or 3D array, optional
            Costs of preventive replacements, by default None.
        rate : float, 2D array or 3D array, optional
            Discount rate, by default None.
        n_samples : int, optional
            Number of samples, by default 1.
        random_state : int, optional
            Random seed, by default None.

        Returns
        -------
        ReplacementPolicyData
            Samples of replacement times, durations, costs and events for each asset.

        Notes
        -----
        If an argument is None, the value of the class attribute is taken.
        """
        (
            model_args,
            reward_args,
            model1_args,
            reward1_args,
            discount_args,
        ) = self.rrp_args(ar, ar1, cf, cp, rate)
        data = self.rrp.sample(
            T,
            model_args,
            reward_args,
            model1_args,
            reward1_args,
            discount_args,
            n_samples,
            random_state,
        )
        events = np.ones(data.size)
        args = args_take(data.indices, *self.args)
        a0 = np.zeros(data.size)
        ind1 = 0
        if self.model1 is not None:
            ar1 = model1_args[0]
            ind1 = data.n_indices * data.n_samples
            a0[:ind1] = args_take(data.indices[:ind1], *self.args1)[0]
            events[:ind1] = np.where(
                data.durations[:ind1] < args_take(data.indices[:ind1], ar1)[0].ravel(),
                1,
                0,
            )
        ar = model_args[0]
        events[ind1:] = np.where(
            data.durations[ind1:] < args_take(data.indices[ind1:], ar)[0].ravel(), 1, 0
        )
        return ReplacementPolicyData(*data.astuple(), events, args, a0)
