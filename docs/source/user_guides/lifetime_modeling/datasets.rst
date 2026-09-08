Datasets
=========

ReLife ships three example datasets under ``relife.datasets``, all about
power-grid equipment lifetime data. Each one is loaded as a NumPy structured array with a
``time`` (observed lifetime), ``event`` (``True`` if the failure was actually observed,
``False`` if right-censored) and ``entry`` (left-truncation age) field; see
:doc:`censoring_and_truncation` for what those mean.

Circuit breakers
------------------

>>> from relife.datasets import load_circuit_breaker
>>> dataset = load_circuit_breaker()
>>> dataset.dtype.names
('time', 'event', 'entry')
>>> len(dataset)
4204

4204 circuit breakers with no covariates. Only a small fraction (204, ~5%) were actually
observed failing (the rest are right-censored), and 4000 were already in service before
entering observation. This heavy censoring makes it a good illustration of
:doc:`non_parametric_models`.

Power transformers
---------------------

>>> from relife.datasets import load_power_transformer
>>> dataset = load_power_transformer()
>>> dataset.dtype.names
('time', 'event', 'entry')
>>> len(dataset)
1650

1650 power transformers with no covariates, a more balanced mix of 318 observed failures and
1158 left-truncated units. Used for parametric distribution fitting throughout the
:doc:`Lifetime modeling <index>` guide.

Insulator strings
--------------------

>>> from relife.datasets import load_insulator_string
>>> dataset = load_insulator_string()
>>> dataset.dtype.names
('time', 'event', 'entry', 'pHCl', 'pH2SO4', 'HNO3')
>>> len(dataset)
12000

12000 insulator strings, the only dataset with covariates: ``pHCl``, ``pH2SO4`` and
``HNO3``, the acid concentrations the insulators are exposed to, which accelerate their
degradation. 2196 observed failures and 8216 left-truncated units. Used for both the
parametric regression and the semi-parametric Cox model in
:doc:`regressions`.
