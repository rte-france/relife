TIMELINE_DOCSTRING = """
Calculate the {name}  over a given timeline.

It takes into account ``discounting_rate`` attribute value.

{formula}

Parameters
----------
timeline: np.ndarray
    Values of the timeline over which the {name} is to be calculated.

Returns
-------
np.ndarray
    The {name}.
"""

ASYMPTOTIC_DOCSTRING = """
Calculate the asymptotic {name}.

It takes into account ``discounting_rate`` attribute value.

{formula}

Returns
-------
np.ndarray
    The asymptotic {name}.
"""

ETC_FORMULA = r"""
The expected total cost :math:`z(t)` is computed by solving the renewal equation and is given by:

.. math::

    z(t) = \mathbb{E}(Z_t) = \int_{0}^{\infty}\mathbb{E}(Z_t~|~X_1 = x)dF(x)

where :

- :math:`t` is the time
- :math:`X_i \sim F` are :math:`n` random variable lifetimes, *i.i.d.*, of cumulative distribution :math:`F`.
- :math:`Z_t` is the random variable reward at each time :math:`t`.
- :math:`\delta` is the discounting rate.
"""

EEAC_FORMULA = r"""
The expected equivalent annual cost :math:`\text{EEAC}(t)` is given by:

.. math::

    \text{EEAC}(t) = \dfrac{\delta z(t)}{1 - e^{-\delta t}}

where :

- :math:`t` is the time
- :math:`z(t)` is the expected_total_cost at :math:`t`. See :py:meth:`~AgeReplacementPolicy.expected_total_cost` for more details.`.
- :math:`\delta` is the discounting rate.
"""

ASYMPTOTIC_ETC_FORMULA = r"""
The asymptotic expected total cost is:
 
.. math::

    \lim_{t\to\infty} z(t)

where :math:`z(t)` is the expected total cost at :math:`t`. See :py:meth:`~AgeReplacementPolicy.expected_total_cost` for more details.
"""

ASYMPTOTIC_EEAC_FORMULA = r"""
The asymptotic expected total cost is:

.. math::

    \lim_{t\to\infty} \text{EEAC}(t)
    
where :math:`\text{EEAC}(t)` is the expected equivalent annual cost at :math:`t`. See :py:meth:`~AgeReplacementPolicy.expected_equivalent_annual_cost` for more details.
"""

ETC_DOCSTRING = TIMELINE_DOCSTRING.format(name="expected total cost", formula=ETC_FORMULA)
EEAC_DOCSTRING = TIMELINE_DOCSTRING.format(name="expected equivalent annual cost", formula=EEAC_FORMULA)
ASYMPTOTIC_ETC_DOCSTRING = ASYMPTOTIC_DOCSTRING.format(name="expected total cost", formula=ASYMPTOTIC_ETC_FORMULA)
ASYMPTOTIC_EEAC_DOCSTRING = ASYMPTOTIC_DOCSTRING.format(
    name="expected equivalent annual cost", formula=ASYMPTOTIC_EEAC_FORMULA
)
