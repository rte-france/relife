From process to policy
=========================

A policy is what turns the :doc:`renewal reward process <reward_framework>` into a
maintenance decision: it fixes the reward shape, gives the methods maintenance names, and
adds whatever is specific to the decision at hand. This page describes what the policies have
in common; :doc:`../lifetime_modeling/run_to_failure` and
:doc:`../lifetime_modeling/preventive_age_replacement` describe each of them
in turn.

Two options for every policy
-------------------------------

Each policy comes in two options: a *renewal* one, where the asset is replaced indefinitely,
which is the right model for planning a fleet over the long run, and a *one-cycle* one, which
stops at the first replacement, the right model for a decision about the asset currently in
service. The two share the API described below, but they do not annualize over the same
horizon since expected cost are annualized over the realized cycle duration rather than
a long-run rate

The four cost methods
-----------------------

Every policy, renewal or one-cycle, run-to-failure or age-replacement, exposes the same four
methods. They are the reward methods of the underlying
:doc:`renewal reward process <reward_framework>`, renamed into maintenance vocabulary:

===============================================  ==============================================
policy                                           ``RenewalRewardProcess``
===============================================  ==============================================
``expected_net_present_value``                   ``expected_total_reward``
``asymptotic_expected_net_present_value``        ``asymptotic_expected_total_reward``
``expected_equivalent_annual_cost``              ``expected_equivalent_annual_worth``
``asymptotic_expected_equivalent_annual_cost``   ``asymptotic_expected_equivalent_annual_worth``
===============================================  ==============================================

The net present value is what the policy will have cost by a given date, counted in money of
today; the equivalent annual cost is that same amount rewritten as a constant yearly payment,
which is what makes two policies comparable. In each pair, the plain method works on an
explicit horizon and takes a final time ``tf`` and a number of points ``nb_steps``, returning
the timeline together with the values along it, while the ``asymptotic_`` one takes no
timeline and returns a single number. Everything else is keyword-only: the costs (``cf``,
plus ``cp`` for the age-replacement policies), the replacement age ``ar``, the initial age
``a0``, and the ``discounting_rate``.

>>> import numpy as np
>>> from relife.datasets import load_circuit_breaker
>>> from relife.lifetime_models import Weibull
>>> from relife.policies import AgeReplacementPolicy
>>> dataset = load_circuit_breaker()
>>> weibull = Weibull().fit(dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> policy = AgeReplacementPolicy(weibull)
>>> timeline, npv = policy.expected_net_present_value(200., 201, ar=47.44, cf=1500., cp=400.)
>>> np.round(npv[::50], 2)
array([   0.  ,  539.97, 1081.47, 1625.66, 2173.95])
>>> timeline, eeac = policy.expected_equivalent_annual_cost(200., 201, ar=47.44, cf=1500., cp=400.)
>>> np.round(eeac[::50], 2)
array([ 0.  , 10.8 , 10.81, 10.84, 10.87])
>>> round(float(policy.asymptotic_expected_equivalent_annual_cost(ar=47.44, cf=1500., cp=400.)), 2)
11.69

The age-replacement policies want both costs together, since ``ar`` describes no decision
without a ``cp`` to weigh against ``cf``:

>>> policy.asymptotic_expected_equivalent_annual_cost(ar=47.44, cf=1500.)
Traceback (most recent call last):
    ...
TypeError: Missing cf and cp values
