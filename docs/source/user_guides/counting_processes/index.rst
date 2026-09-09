Counting processes
===================

Counting processes cover **repairable** assets: the same asset fails several times and each
repair is *minimal*, meaning it restores service without rejuvenating the asset. What is
modeled is the rate of recurrent failures rather than a duration. ReLife implements the
non-homogeneous Poisson process (:py:class:`~relife.stochastic_processes.NonHomogeneousPoissonProcess`)
and the replacement policy built on it
(:py:class:`~relife.policies.NonHomogeneousPoissonAgeReplacementPolicy`).

.. toctree::
    :maxdepth: 1

    datasets
    nhpp
    maintenance_policies
    sampling
