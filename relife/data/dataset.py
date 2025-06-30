from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_power_transformer() -> NDArray[np.void]:
    r"""
    Load power transformer dataset containing observed lifetimes and left truncations values.

    Examples
    --------
    >>> from relife.data import load_power_transformer
    >>> data = load_power_transformer()
    >>> print(data["time"])
    [34.3 45.1 53.2 ... 30.  30.  30. ]

    Returns
    -------
    structured array
        A numpy structured array of 3 fields :

        - time (``np.float64``) : observed lifetime values
        - event (``np.bool_``) : boolean flag indicated if the event has been observed or not (if False, the observed lifetimes are right censored)
        - entry (``np.float64``) : left truncation values

    """

    data = np.loadtxt(
        Path(Path(__file__).parents[0], "csv/power_transformer.csv"),
        delimiter=",",
        skiprows=1,
        dtype=np.dtype([("time", np.float64), ("event", np.float64), ("entry", np.float64)]),
    )
    # for some reason, numpy can't cast 1.0/0.0 to np.bool_
    new_dtype = np.dtype([("time", np.float64), ("event", np.bool_), ("entry", np.float64)])
    return data.astype(new_dtype)


def load_insulator_string() -> NDArray[np.void]:
    r"""
    Load insulator string dataset containing observed lifetimes, left truncations values and covariates

    Examples
    --------
    >>> from relife.data import load_insulator_string
    >>> data = load_insulator_string()
    >>> print(data["time"])
    [70.  30.  45.  ...  8.8  7.6 53. ]
    >>> print(data["pHCl"])
    [0.49 0.76 0.43 ... 1.12 1.19 0.35]

    Returns
    -------
    structured array
        A numpy structured array of 3 fields :

        - time (``np.float64``) : observed lifetime values
        - event (``np.bool_``) : boolean flag indicated if the event has been observed or not (if False, the observed lifetimes are right censored)
        - entry (``np.float64``) : left truncation values
        - pHCl (``np.float64``) : quantitative covariate values (concentration of pHCl)
        - pH2SO4 (``np.float64``) : quantitative covariate values (concentration of pH2SO4)
        - HNO3 (``np.float64``) : quantitative covariate values (concentration of HNO3)

    """

    data = np.loadtxt(
        Path(Path(__file__).parents[0], "csv/insulator_string.csv"),
        delimiter=",",
        skiprows=1,
        dtype=np.dtype(
            [
                ("time", np.float64),
                ("event", np.float64),
                ("entry", np.float64),
                ("pHCl", np.float64),
                ("pH2SO4", np.float64),
                ("HNO3", np.float64),
            ]
        ),
    )
    # for some reason, numpy can't cast 1.0/0.0 to np.bool_
    new_dtype = np.dtype(
        [
            ("time", np.float64),
            ("event", np.bool_),
            ("entry", np.float64),
            ("pHCl", np.float64),
            ("pH2SO4", np.float64),
            ("HNO3", np.float64),
        ]
    )
    return data.astype(new_dtype)


def load_circuit_breaker() -> NDArray[np.void]:
    r"""
    Load circuit breaker dataset containing observed lifetimes and left truncations values

    Examples
    --------
    >>> from relife.data import load_circuit_breaker
    >>> data = load_circuit_breaker()
    >>> print(data["time"])
    [34. 28. 12. ... 42. 42. 37.]

    Returns
    -------
    structured array
        A numpy structured array of 3 fields :

        - time (``np.float64``) : observed lifetime values
        - event (``np.bool_``) : boolean flag indicated if the event has been observed or not (if False, the observed lifetimes are right censored)
        - entry (``np.float64``) : left truncation values
    """

    data = np.loadtxt(
        Path(Path(__file__).parents[0], "csv/circuit_breaker.csv"),
        delimiter=",",
        skiprows=1,
        dtype=np.dtype([("time", np.float64), ("event", np.float64), ("entry", np.float64)]),
    )
    # for some reason, numpy can't cast 1.0/0.0 to np.bool_
    new_dtype = np.dtype([("time", np.float64), ("event", np.bool_), ("entry", np.float64)])
    return data.astype(new_dtype)


# def load_input_turnbull() -> NDArray[np.void]:
#     """_summary_
#
#     Returns:
#         np.ndarray: _description_
#     """
#     data = np.loadtxt(
#         Path(Path(__file__).parents[0], "csv/input_turnbull.csv"),
#         delimiter=",",
#         skiprows=1,
#     )
#     data["event"] = data["event"].astype(np.bool_)
#     return data
