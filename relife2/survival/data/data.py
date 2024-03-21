from itertools import combinations
from typing import Tuple, Union

import numpy as np

from ..errors import LifetimeDataError
from ..errors.checks import is_array
from .parsers import (
    ParsedData,
    complete_factory,
    interval_censored_factory,
    interval_truncated_factory,
    left_censored_factory,
    left_truncated_factory,
    right_censored_factory,
    right_truncated_factory,
)


class Data:
    """
    Args:
        observed_lifetimes (np.ndarray): observed lifetimes values. Shape must be (n,)
        complete_indicators (np.ndarray, optional): _description_. Defaults to None.
        left_censored_indicators (np.ndarray, optional): _description_. Defaults to None.
        right_censored_indicators (np.ndarray, optional): _description_. Defaults to None.
        entry (np.ndarray, optional): _description_. Defaults to None.
        departure (np.ndarray, optional): _description_. Defaults to None.
    """

    def __init__(
        self,
        observed_lifetimes: np.ndarray,
        complete_indicators: np.ndarray = None,
        left_censored_indicators: np.ndarray = None,
        right_censored_indicators: np.ndarray = None,
        entry: np.ndarray = None,
        departure: np.ndarray = None,
    ):

        try:
            is_array(observed_lifetimes)
        except Exception as error:
            raise LifetimeDataError(
                "Invalid observed_lifetimes data type"
            ) from error

        self.left_censored = left_censored_factory(
            observed_lifetimes, indicators=left_censored_indicators
        )

        self.right_censored = right_censored_factory(
            observed_lifetimes, indicators=right_censored_indicators
        )

        if (
            left_censored_indicators is not None
            and right_censored_indicators is not None
        ):
            if len(left_censored_indicators) != len(right_censored_indicators):
                raise LifetimeDataError(
                    """
                    Expected left_censored_indicators and right_censored_indicators
                    to have the same length
                    """
                )

            if (
                np.stack(
                    [left_censored_indicators, right_censored_indicators]
                ).any(axis=0)
                is True
            ):
                raise LifetimeDataError(
                    """
                    left_censored_indicators and right_censored_indicators
                    can't be true at the same index
                    """
                )

        self.interval_censored = interval_censored_factory(observed_lifetimes)

        if left_censored_indicators:
            complete_indicators = np.stack([not left_censored_indicators])
            if right_censored_indicators:
                complete_indicators = np.stack(
                    [complete_indicators, not right_censored_indicators]
                )
            complete_indicators = complete_indicators.all(axis=0)

        self.complete = complete_factory(
            observed_lifetimes, indicators=complete_indicators
        )

        if entry is not None:
            if len(entry) != len(observed_lifetimes):
                raise LifetimeDataError(
                    """
                    entry values (left truncation values) and observed_lifetimes
                    don't have the same length
                    """
                )
        if departure is not None:
            if len(departure) != len(observed_lifetimes):
                raise LifetimeDataError(
                    """
                    departure values (right truncation values) and observed_lifetimes
                    don't have the same length
                    """
                )

        self.left_truncated = left_truncated_factory(entry, departure)
        self.right_truncated = right_truncated_factory(entry, departure)
        self.interval_truncated = interval_truncated_factory(
            entry,
            departure,
        )

        field_names = [
            "complete",
            "left_censored",
            "right_censored",
            "interval_censored",
            "right_truncated",
            "left_truncated",
            "interval_truncated",
        ]
        self._intersection_data = {}

        # populate with all possible combinations of data
        for n in range(1, len(field_names)):
            for combination in combinations(field_names, n):
                sorted_combination = sorted(combination)
                if Data.check_field_combination(sorted_combination):
                    self._intersection_data[
                        " & ".join(sorted_combination)
                    ] = self._intersection(*sorted_combination)
                    # print(
                    #     "[VALID]: ",
                    #     sorted_combination,
                    #     self._intersection(*sorted_combination),
                    # )
                # if field combinations is not allowed, check if no data exists
                else:
                    extracted_data = self._intersection(*sorted_combination)
                    if {len(_data) for _data in extracted_data} != {0}:
                        raise LifetimeDataError(
                            f"""
                            Incoherence in provided data :
                            {sorted_combination} data is impossible
                            """
                        )

        # compute other checks on data values (TO BE COMPLETED)
        self._sanity_checks()

    def _sanity_checks(self):

        if (
            self.interval_censored.values[:, 0]
            >= self.interval_censored.values[:, 1]
        ).any():
            raise LifetimeDataError("Invalid interval censorship values")
        if (
            self.interval_truncated.values[:, 0]
            >= self.interval_truncated.values[:, 1]
        ).any():
            raise LifetimeDataError("Invalid interval truncation values")

        # print(self.values("right_censored & left_truncated"))
        extracted_data = self("right_censored & left_truncated")
        censored_values, truncation_values = (
            extracted_data[0].values,
            extracted_data[1].values,
        )
        # print(censored_values)
        # print(truncation_values)
        if (censored_values <= truncation_values).any():
            raise LifetimeDataError(
                f"""
                right censored lifetime values can't be lower or equal to left
                truncation values:
                incompatible {censored_values} and {truncation_values}
                """
            )

        # print(self.values("right_censored & interval_truncated"))
        extracted_data = self("right_censored & interval_truncated")
        censored_values, truncation_values = (
            extracted_data[0].values,
            extracted_data[1].values,
        )
        if (censored_values >= truncation_values[:, 1]).any():
            raise LifetimeDataError(
                f"""
                right censored lifetime values can't be higer or equal to interval
                of truncation: incompatible {censored_values} and {truncation_values}
                """
            )

        extracted_data = self("interval_censored & interval_truncated")
        censored_values, truncation_values = (
            extracted_data[0].values,
            extracted_data[1].values,
        )
        if (censored_values[:, 0] < truncation_values[:, 0]).any():
            raise LifetimeDataError(
                f"""
                interval censorship can't be outside of truncation interval:
                incompatible {censored_values} and {truncation_values}
                """
            )
        elif (censored_values[:, 1] > truncation_values[:, 1]).any():
            raise LifetimeDataError(
                f"""
                interval censorship can't be outside of truncation interval:
                incompatible {censored_values} and {truncation_values}
                """
            )

        extracted_data = self("right_censored & right_truncated")
        censored_values, truncation_values = (
            extracted_data[0].values,
            extracted_data[1].values,
        )
        if (censored_values >= truncation_values).any():
            raise LifetimeDataError(
                f"""
                right censored lifetime values can't be higher than right truncations:
                incompatible {censored_values} and {truncation_values}
                """
            )

        extracted_data = self("left_censored & right_truncated")
        censored_values, truncation_values = (
            extracted_data[0].values,
            extracted_data[1].values,
        )
        if (censored_values >= truncation_values).any():
            raise LifetimeDataError(
                f"""
                left censored lifetime values can't be higher than right truncations:
                incompatible {censored_values} and {truncation_values}
                """
            )

        extracted_data = self("interval_censored & right_truncated")
        censored_values, truncation_values = (
            extracted_data[0].values,
            extracted_data[1].values,
        )
        if (censored_values[:, 1] >= truncation_values).any():
            raise LifetimeDataError(
                f"""
                interval censored lifetime values can't be higher than right truncations
                : incompatible {censored_values} and {truncation_values}
                """
            )

        extracted_data = self("complete & right_truncated")
        observed_values, truncation_values = (
            extracted_data[0].values,
            extracted_data[1].values,
        )
        if (observed_values >= truncation_values).any():
            raise LifetimeDataError(
                f"""
                complete lifetime values can't be higher than right truncations:
                incompatible {observed_values} and {truncation_values}
                """
            )

        extracted_data = self("complete & left_truncated")
        observed_values, truncation_values = (
            extracted_data[0].values,
            extracted_data[1].values,
        )
        if (observed_values <= truncation_values).any():
            raise LifetimeDataError(
                f"""
                complete lifetime values can't be lower than left truncations:
                incompatible {observed_values} and {truncation_values}
                """
            )

        extracted_data = self("complete & interval_truncated")
        observed_values, truncation_values = (
            extracted_data[0].values,
            extracted_data[1].values,
        )
        if (observed_values >= truncation_values[:, 1]).any():
            raise LifetimeDataError(
                f"""
                complete lifetime values can't be outside of truncation interval:
                incompatible {observed_values} and {truncation_values}
                """
            )
        elif (observed_values <= truncation_values[:, 0]).any():
            raise LifetimeDataError(
                f"""
                complete lifetime values can't be outside of truncation interval:
                incompatible {observed_values} and {truncation_values}"""
            )

    @staticmethod
    def check_field_combination(fields: list) -> bool:
        def other_not_allowed(x):
            if x == "complete":
                return [
                    "complete",
                    "left_censored",
                    "right_censored",
                    "interval_censored",
                ]
            elif x == "left_censored":
                return [
                    "left_censored",
                    "complete",
                    "right_censored",
                    "interval_censored",
                    "left_truncated",
                    "interval_truncated",
                ]
            elif x == "right_censored":
                return [
                    "right_censored",
                    "complete",
                    "interval_censored",
                    "left_censored",
                ]
            elif x == "interval_censored":
                return [
                    "interval_censored",
                    "complete",
                    "left_censored",
                    "right_censored",
                ]
            elif x == "left_truncated":
                return [
                    "left_truncated",
                    "interval_truncated",
                    "left_censored",
                    "right_truncated",
                    "interval_censored",
                ]
            elif x == "right_truncated":
                return [
                    "right_truncated",
                    "interval_truncated",
                    "left_truncated",
                ]
            elif x == "interval_truncated":
                return [
                    "interval_truncated",
                    "left_truncated",
                    "right_truncated",
                    "left_censored",
                ]

        not_allowed = []
        res = True
        for i in range(len(fields) - 1):
            cursor = fields[i]
            rest = fields[i + 1 :]
            not_allowed += other_not_allowed(cursor)
            if len(set(rest).intersection(not_allowed)) != 0:
                res = False
                break
        return res

    def _intersection(self, *fields: str) -> Tuple[ParsedData]:
        assert {type(field) for field in fields} == {
            str
        }, "intersection expects string arguments"
        assert {hasattr(self, field) for field in fields} == {
            True
        }, f"field names {fields} are unknown"

        def join_index(*index: np.ndarray):
            s = set.intersection(*map(set, index))
            mask_index = [
                np.in1d(_index, np.array(list(s))) for _index in index
            ]
            index = index[0][mask_index[0]]
            return index, mask_index

        index = [getattr(self, f"{field}").index for field in fields]
        values = [getattr(self, f"{field}").values for field in fields]

        common_index, mask_index = join_index(*index)
        return tuple(
            (
                ParsedData(common_index, values[i][mask])
                for i, mask in enumerate(mask_index)
            )
        )

    def __call__(self, request: str) -> Union[ParsedData, Tuple[ParsedData]]:
        def isort(x: list):
            return sorted(range(len(x)), key=lambda k: x[k])

        request = "".join(request.split())

        if ("&" in request) and ("|" not in request):
            fields = request.split("&")
            sorted_fields = sorted(fields)
            if {hasattr(self, field) for field in fields} != {True}:
                raise ValueError(f"field names {fields} are unknown")
            if not " & ".join(sorted_fields) in self._intersection_data:
                raise ValueError(
                    f"impossible combination of fields : {fields}"
                )

            sorted_fields_i = isort(fields)
            return tuple(
                list(self._intersection_data[" & ".join(sorted_fields)])[i]
                for i in sorted_fields_i
            )

        elif ("|" in request) and ("&" not in request):
            fields = request.split("|")
            if {hasattr(self, field) for field in fields} != {True}:
                raise ValueError(f"field names {fields} are unknown")
            return tuple(
                (
                    ParsedData(
                        getattr(self, f"{field}").index,
                        getattr(self, f"{field}").values,
                    )
                    for field in fields
                )
            )
        elif ("|" in request) and ("&" in request):
            raise ValueError("can't hold & and | operator")
        else:
            if not hasattr(self, request):
                raise ValueError(f"field name {request} is unknown")
            return ParsedData(
                getattr(self, f"{request}").index,
                getattr(self, f"{request}").values,
            )

    def __len__(self):
        return (
            len(self.complete)
            + len(self.left_censored)
            + len(self.right_censored)
            + len(self.interval_censored)
            + len(self.left_truncated)
            + len(self.right_truncated)
            + len(self.interval_truncated)
        )

    def info(self) -> None:
        headers = ["Lifetime data", "Counts"]
        info = []
        info.append(["Nb samples (tot.)", len(self)])
        info.append(["Observed", len(self.complete)])
        info.append(["Left censored", len(self.left_censored)])
        info.append(["Right censored", len(self.right_censored)])
        info.append(["Interval censored", len(self.interval_censored)])
        info.append(["Left truncated", len(self.left_truncated)])
        info.append(["Right truncated", len(self.right_truncated)])
        info.append(["Interval truncated", len(self.interval_truncated)])

        row_format = "{:>18}" * (len(headers))
        print(row_format.format(*headers))
        for row in info:
            print(row_format.format(*row))
