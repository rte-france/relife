from dataclasses import dataclass, field
from typing import Literal, Unpack

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from typing_extensions import overload

from relife.likelihood._base import FittingResults
from relife.likelihood._partial_lifetime_likelihood import (
    BreslowPartialLifetimeLikelihood,
    CoxPartialLifetimeLikelihood,
    EfronPartialLifetimeLikelihood,
)
from relife.typing import ScipyMinimizeOptions
from relife.utils import reshape_1d_arg

from ._regression import LinearCovarEffect


@dataclass
class CoxData:
    # TODO: à compléter
    """
    Object that encapsultates data used in Cox model estimation and inference.

    Attributes
    ----------
    ...
    """

    time: NDArray[np.float64]
    covar: NDArray[np.float64]
    event: NDArray[np.bool_] | None = None
    entry: NDArray[np.float64] | None = None
    likelihood_to_use: Literal["cox"] | Literal["breslow"] | Literal["efron"] = field(
        init=False, repr=True
    )
    event_count: NDArray[np.int64] = field(init=False, repr=False)
    risk_set: NDArray[np.bool_] = field(init=False, repr=False)
    death_set: NDArray[np.bool_] = field(init=False, repr=False)
    ordered_event_covar: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        self.time = reshape_1d_arg(self.time)
        self.event = (
            reshape_1d_arg(self.event)
            if self.event is not None
            else np.ones_like(self.time, dtype=np.bool_)
        )
        self.entry = (
            reshape_1d_arg(self.entry)
            if self.entry is not None
            else np.zeros_like(self.time, dtype=np.float64)
        )
        sizes = [len(x) for x in (self.time, self.event, self.entry, self.covar)]

        if len(set(sizes)) != 1:
            raise ValueError(
                f""""
                All lifetime data must have the same number of values. Fields
                length are different. Got {tuple(sizes)}.
                """
            )
        (
            ordered_event_time,  # uncensored sorted untied times
            ordered_event_index,
            self.event_count,
        ) = np.unique(
            self.time[self.event == 1],
            return_index=True,
            return_counts=True,
        )
        # here risk_set is mask array on time
        # left truncated & right censored
        self.risk_set = np.logical_and(
            (
                np.vstack([self.entry[:, 0]] * len(ordered_event_time))
                < np.hstack([ordered_event_time[:, None]] * len(self.time))
            ),
            (
                np.hstack([ordered_event_time[:, None]] * len(self.time))
                <= np.vstack([self.time[:, 0]] * len(ordered_event_time))
            ),
        )

        self.death_set = np.vstack(
            [self.time[:, 0] * self.event[:, 0]] * len(ordered_event_time)
        ) == np.hstack([ordered_event_time[:, None]] * len(self.time))

        self.ordered_event_covar = self.covar[self.event[:, 0] == 1][
            ordered_event_index
        ]


def psi(
    covar_effect: LinearCovarEffect,
    data: CoxData,
    on: Literal["risk"] | Literal["death"] = "risk",
    order: Literal[0] | Literal[1] | Literal[2] = 0,
) -> NDArray[np.float64]:
    """Psi formula used for likelihood computations

    Args:
        on (str, optional): "risk" or "death". Defaults to "risk". If "death",
        sum is applied on death set. order (int, optional): order derivatives
        with respect to params. Defaults to 0.

    Returns:
        np.ndarray: psi formulation
        If order 0, shape [m, 1]
        If order 1, shape [m, p]
        If order 2, shape [m, p, p]
    """
    if on == "risk":
        i_set = data.risk_set
    elif on == "death":
        i_set = data.death_set

    if order == 0:
        # shape [m]
        return np.dot(i_set, covar_effect.g(data.covar))
    elif order == 1:
        # shape [m, p]
        return np.dot(i_set, data.covar * covar_effect.g(data.covar))
    elif order == 2:
        # shape [m, p, p]
        return np.tensordot(
            i_set[:, :None],
            data.covar[:, None]
            * data.covar[:, :, None]
            * np.asarray(covar_effect.g(data.covar))[:, :, None],
            axes=1,
        ).astype(np.float64)


class _BreslowBaseline:
    """
    Class for Cox non-parametric Breslow baseline
    """

    data: CoxData
    covar_effect: LinearCovarEffect

    def __init__(self, data: CoxData, covar_effect: LinearCovarEffect):
        assert data.covar.shape[-1] == covar_effect.nb_params
        self.covar_effect = covar_effect
        self.data = data

    @overload
    def chf(
        self, conf_int: Literal[False], kp: bool = False
    ) -> NDArray[np.float64]: ...
    @overload
    def chf(
        self, conf_int: Literal[True], kp: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def chf(
        self, conf_int: bool = False, kp: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | NDArray[np.float64]:
        # TODO : docstring

        # TODO : comprendre le kp
        if kp:
            values = np.cumsum(
                1
                - (
                    1
                    - (
                        self.covar_effect.g(self.data.ordered_event_covar)
                        / psi(self.covar_effect, self.data)
                    )
                )
                ** (self.covar_effect.g(self.data.ordered_event_covar))
            )
        else:
            values = np.cumsum(
                self.data.event_count[:, None] / psi(self.covar_effect, self.data)
            )
        if conf_int:
            var = np.cumsum(
                self.data.event_count[:, None] / psi(self.covar_effect, self.data) ** 2
            )
            conf_int_values = np.hstack(
                [
                    values[:, None]
                    + np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                    values[:, None]
                    - np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                ]
            )
            return values, conf_int_values
        else:
            return values

    @overload
    def sf(self, conf_int: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def sf(
        self, conf_int: Literal[True]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def sf(
        self, conf_int: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | NDArray[np.float64]:
        # TODO : docstring

        if conf_int:
            chf, chf_conf_int_values = self.chf(conf_int=True)
            return np.exp(-chf), np.exp(-chf_conf_int_values)
        else:
            return np.exp(-self.chf(conf_int=False))


class SemiParametricProportionalHazard:
    """
    Class for Cox, semi-parametric, Proportional Hazards, model
    """

    fitting_results: FittingResults | None
    covar_effect: LinearCovarEffect | None
    _sf: NDArray[np.void] | None

    def __init__(self):
        self._sf = None
        self.fitting_results = None
        self.covar_effect = None

    @property
    def params(self):
        if self.covar_effect is None:
            return None
        return self.covar_effect.params

    @property
    def nb_params(self):
        if self.covar_effect is None:
            return None
        return self.covar_effect.nb_params

    @overload
    def sf(
        self, se: Literal[False]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def sf(
        self, se: Literal[True]
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
    ): ...
    @overload
    def sf(
        self, se: bool
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        | None
    ): ...

    def sf(
        self, se: bool = True
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        | None
    ):
        """
        The survival function estimations.

        Parameters
        ----------
        se : bool, default True
            If True, the standard errors are returned in addition to timeline
            and sf values.

        Returns
        -------
        out : tuple of timeline, values, optionally se. Default is None
            A timeline, corresponding sf values and optionnaly the standard
            errors. If the estimations does not exist yet, returns None.
        """
        if self._sf is None:
            return None
        if se:
            return self._sf["timeline"], self._sf["estimation"], self._sf["se"]
        return self._sf["timeline"], self._sf["estimation"]

    def fit(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        **optimizer_options: Unpack[ScipyMinimizeOptions],
    ):
        # init covar_effect
        self.covar_effect = LinearCovarEffect(
            (None,) * np.atleast_2d(np.asarray(covar, dtype=np.float64)).shape[-1]
        )
        # encapsulate data and computes other informations (ordered_event_time, etc.)
        data = CoxData(time, covar, event=event, entry=entry)
        # data_nhpp = ...
        # data_nhpp.to_lifetime_data() -> LifetimeData

        if (data.event_count > 3).any():  # efron
            likelihood = EfronPartialLifetimeLikelihood(self.covar_effect, data)
        elif (data.event_count <= 3).all() and (2 in data.event_count):  # breslow
            likelihood = BreslowPartialLifetimeLikelihood(self.covar_effect, data)
        else:
            likelihood = CoxPartialLifetimeLikelihood(self.covar_effect, data)

        fitting_results = likelihood.maximum_likelihood_estimation(**optimizer_options)
        self.fitting_results = fitting_results
        self.covar_effect.params = fitting_results.optimal_params.copy()

        # currently only BreslowBaseline is used to compute sf
        baseline = _BreslowBaseline(data, self.covar_effect)

        # estimate sf
        values = baseline.sf(conf_int=False) ** self.covar_effect.g(covar)

        if fitting_results.covariance_matrix is not None:
            psi_values = psi(self.covar_effect, data)
            psi_order_1 = psi(self.covar_effect, data, order=1)
            d_j_on_psi = data.event_count[:, None] / psi_values

            q3 = np.cumsum(
                (psi_order_1 / psi_values - covar) * d_j_on_psi, axis=0
            )  # [m, p]
            q2 = np.squeeze(
                np.matmul(
                    q3[:, None, :],
                    np.matmul(
                        fitting_results.covariance_matrix[None, :, :],
                        q3[:, :, None],
                    ),
                )
            )  # m
            q1 = np.cumsum(d_j_on_psi * (1 / psi_values))

            var = (values**2) * (q1 + q2)

            conf_int = np.hstack(
                [
                    values[:, None]
                    + np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                    values[:, None]
                    - np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                ]
            )

        # TODO: à compléter
        # voir _non_parametric pour exemple
        # pass values, conf_int in self._sf
        # self._sf = ...
        return self
