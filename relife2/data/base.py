from dataclasses import dataclass

import numpy as np

from .parser import (
    Data,
    IntervalData,
    interval_censored_factory,
    interval_truncated_factory,
    left_censored_factory,
    left_truncated_factory,
    observed_factory,
    right_censored_factory,
    right_truncated_factory,
)


@dataclass
class SurvivalData:
    observed: Data  # object with fixed index and values attrib format
    left_censored: Data
    right_censored: Data
    interval_censored: IntervalData
    left_truncated: Data
    right_truncated: Data
    interval_truncated: IntervalData

    def __post_init__(self):
        if (
            self.interval_censored.values[:, 0] >= self.interval_censored.values[:, 1]
        ).any():
            raise ValueError("Invalid interval censorship values")
        if (
            self.interval_truncated.values[:, 0] >= self.interval_truncated.values[:, 1]
        ).any():
            raise ValueError("Invalid interval truncation values")

        self._intersection_data = {"index": {}, "values": {}}

        res = self._censored_and_truncated(how=("left", "left"))
        if res["index"].size != 0:
            raise ValueError(
                "Left censored lifetimes can't be left truncated lifetimes too"
            )
        res = self._censored_and_truncated(how=("left", "interval"))
        if res["index"].size != 0:
            raise ValueError(
                "Left censored lifetimes can't be interval truncated lifetimes too"
            )

        res = self._censored_and_truncated(how=("right", "left"))
        censored_values, truncation_values = res["values"]
        if (censored_values <= truncation_values).any():
            raise ValueError(
                f"right censored lifetime values can't be lower or equal to left truncation values: incompatible {censored_values} and {truncation_values}"
            )
        else:
            self._intersection_data["index"]["right_censored_left_truncated"] = res[
                "index"
            ]
            self._intersection_data["values"]["right_censored_left_truncated"] = res[
                "values"
            ]

        res = self._censored_and_truncated(how=("interval", "left"))
        censored_values, truncation_values = res["values"]
        if (censored_values[:, 0] <= truncation_values).any():
            raise ValueError(
                f"interval censored lifetime values can't be lower or equal to left truncation values: incompatible {censored_values} and {truncation_values}"
            )
        elif (censored_values[:, 1] <= truncation_values).any():
            raise ValueError(
                f"interval censored lifetime values can't be lower or equal to left truncation values: incompatible {censored_values} and {truncation_values}"
            )
        else:
            self._intersection_data["index"]["interval_censored_left_truncated"] = res[
                "index"
            ]
            self._intersection_data["values"]["interval_censored_left_truncated"] = res[
                "values"
            ]

        res = self._censored_and_truncated(how=("right", "interval"))
        censored_values, truncation_values = res["values"]
        if (censored_values >= truncation_values[:, 1]).any():
            raise ValueError(
                f"right censored lifetime values can't be higer or equal to interval of truncation: incompatible {censored_values} and {truncation_values}"
            )
        else:
            self._intersection_data["index"]["right_censored_interval_truncated"] = res[
                "index"
            ]
            self._intersection_data["values"][
                "right_censored_interval_truncated"
            ] = res["values"]

        res = self._censored_and_truncated(how=("interval", "interval"))
        censored_values, truncation_values = res["values"]
        if (censored_values[:, 0] >= truncation_values[:, 0]).any():
            raise ValueError(
                f"interval censorship can't be outside of truncation interval: incompatible {censored_values} and {truncation_values}"
            )
        elif (censored_values[:, 1] <= truncation_values[:, 1]).any():
            raise ValueError(
                f"interval censorship can't be outside of truncation interval: incompatible {censored_values} and {truncation_values}"
            )
        else:
            self._intersection_data["index"][
                "interval_censored_interval_truncated"
            ] = res["index"]
            self._intersection_data["values"][
                "interval_censored_interval_truncated"
            ] = res["values"]

        res = self._censored_and_truncated(how=("right", "right"))
        censored_values, truncation_values = res["values"]
        if (censored_values >= truncation_values).any():
            raise ValueError(
                f"right censored lifetime values can't be higher than right truncations: incompatible {censored_values} and {truncation_values}"
            )
        else:
            self._intersection_data["index"]["right_censored_right_truncated"] = res[
                "index"
            ]
            self._intersection_data["values"]["right_censored_right_truncated"] = res[
                "values"
            ]

        res = self._censored_and_truncated(how=("left", "right"))
        censored_values, truncation_values = res["values"]
        if (censored_values >= truncation_values).any():
            raise ValueError(
                f"left censored lifetime values can't be higher than right truncations: incompatible {censored_values} and {truncation_values}"
            )
        else:
            self._intersection_data["index"]["left_censored_right_truncated"] = res[
                "index"
            ]
            self._intersection_data["values"]["left_censored_right_truncated"] = res[
                "values"
            ]

        res = self._censored_and_truncated(how=("interval", "right"))
        censored_values, truncation_values = res["values"]
        if (censored_values[:, 1] >= truncation_values).any():
            raise ValueError(
                f"interval censored lifetime values can't be higher than right truncations: incompatible {censored_values} and {truncation_values}"
            )
        else:
            self._intersection_data["index"]["interval_censored_right_truncated"] = res[
                "index"
            ]
            self._intersection_data["values"][
                "interval_censored_right_truncated"
            ] = res["values"]

        res = self._observed_and_truncated(how="right")
        observed_values, truncation_values = res["values"]
        if (observed_values >= truncation_values).any():
            raise ValueError(
                f"observed lifetime values can't be higher than right truncations: incompatible {observed_values} and {truncation_values}"
            )
        else:
            self._intersection_data["index"]["observed_right_truncated"] = res["index"]
            self._intersection_data["values"]["observed_right_truncated"] = res[
                "values"
            ]

        res = self._observed_and_truncated(how="left")
        observed_values, truncation_values = res["values"]
        if (observed_values <= truncation_values).any():
            raise ValueError(
                f"observed lifetime values can't be lower than left truncations: incompatible {observed_values} and {truncation_values}"
            )
        else:
            self._intersection_data["index"]["observed_left_truncated"] = res["index"]
            self._intersection_data["values"]["observed_left_truncated"] = res["values"]

        res = self._observed_and_truncated(how="interval")
        observed_values, truncation_values = res["values"]
        if (observed_values >= truncation_values[:, 1]).any():
            raise ValueError(
                f"observed lifetime values can't be outside of truncation interval: incompatible {observed_values} and {truncation_values}"
            )
        elif (observed_values <= truncation_values[:, 0]).any():
            raise ValueError(
                f"observed lifetime values can't be outside of truncation interval: incompatible {observed_values} and {truncation_values}"
            )
        else:
            self._intersection_data["index"]["observed_interval_truncated"] = res[
                "index"
            ]
            self._intersection_data["values"]["observed_interval_truncated"] = res[
                "values"
            ]

    def _censored_and_truncated(self, how=("right", "left")):
        """return censored and truncated (simultaneously) lifetimes index or corresponding censorship values and truncation values

        Args:
            how (tuple, optional): _description_. Defaults to ("right", "left").
            return_values (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        how_0 = how[0]
        how_1 = how[1]
        censored_index = getattr(self, f"{how_0}_censored").index
        truncated_index = getattr(self, f"{how_1}_truncated").index
        index_censored_and_truncated = truncated_index[
            np.in1d(truncated_index, censored_index)
        ]

        censored_lifetime_values = getattr(self, f"{how_0}_censored").values
        truncation_values = getattr(self, f"{how_1}_truncated").values
        res_1 = censored_lifetime_values[np.in1d(censored_index, truncated_index)]
        res_2 = truncation_values[np.in1d(truncated_index, censored_index)]

        return {"index": index_censored_and_truncated, "values": (res_1, res_2)}

    def _observed_and_truncated(self, how="left"):
        """return observed and truncated (simultaneously) lifetimes index or corresponding observed values and truncation values

        Args:
            how (str, optional): _description_. Defaults to "left".
            return_values (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        truncated_index = getattr(self, f"{how}_truncated").index
        observed_index = self.observed.index
        index_observed_and_truncated = observed_index[
            np.in1d(observed_index, truncated_index)
        ]

        observed_lifetime_values = self.observed.values
        truncation_values = getattr(self, f"{how}_truncated").values
        res_1 = observed_lifetime_values[np.in1d(observed_index, truncated_index)]
        res_2 = truncation_values[np.in1d(truncated_index, observed_index)]

        return {"index": index_observed_and_truncated, "values": (res_1, res_2)}

    def values(self, arg: str):
        if type(arg) != str:
            raise TypeError("values expects string argument")
        if not hasattr(self, arg):
            raise ValueError(f"SurvivalData does not have {arg} attribute")
        return getattr(self, f"{arg}").values

    def index(self, arg: str):
        if type(arg) != str:
            raise TypeError("index expects string argument")
        if not hasattr(self, arg):
            raise ValueError(f"SurvivalData does not have {arg} attribute")
        return getattr(self, f"{arg}").index

    def intersection_values(self, *args: str):
        if {type(arg) for arg in args} != {str}:
            raise TypeError("intersection expects string arguments")
        if len(args) != 2:
            raise ValueError(
                f"intersection takes 2 arguments, {args} provides too many"
            )
        if not hasattr(self, args[0]):
            invalid_arg = args[0]
            raise ValueError(f"SurvivalData does not have {invalid_arg} attribute")
        if not hasattr(self, args[1]):
            invalid_arg = args[1]
            raise ValueError(f"SurvivalData does not have {invalid_arg} attribute")
        if "observed" in args[0] and "censored" in args[1]:
            raise ValueError(
                f"Invalid intersection args {args} : lifetimes can't be observed and censored at the same time"
            )
        if "censored" in args[0] and "observed" in args[1]:
            raise ValueError(
                f"Invalid intersection args {args} : lifetimes can't be observed and censored at the same time"
            )
        if args[0] == "left_censored" and args[1] == "interval_truncated":
            raise ValueError(
                "Left censored lifetimes can't be interval truncated lifetimes too"
            )
        if args[1] == "left_censored" and args[0] == "interval_truncated":
            raise ValueError(
                "Left censored lifetimes can't be interval truncated lifetimes too"
            )
        if args[0] == "left_censored" and args[1] == "left_truncated":
            raise ValueError(
                "Left censored lifetimes can't be left truncated lifetimes too"
            )
        if args[1] == "left_censored" and args[0] == "left_truncated":
            raise ValueError(
                "Left censored lifetimes can't be left truncated lifetimes too"
            )
        if args[0] == args[1]:
            raise ValueError(
                f"Invalid intersection args {args} : args cannot be equals"
            )
        if "censored" in args[0] and "truncated" in args[1]:
            return self._intersection_data["values"][f"{args[0]}_{args[1]}"]
        if "truncated" in args[0] and "censored" in args[1]:
            return (
                self._intersection_data["values"][f"{args[1]}_{args[0]}"][1],
                self._intersection_data["values"][f"{args[1]}_{args[0]}"][0],
            )
        if args[0] == "observed" and "truncated" in args[1]:
            return self._intersection_data["values"][f"{args[0]}_{args[1]}"]
        if "truncated" in args[0] and args[1] == "observed":
            return (
                self._intersection_data["values"][f"{args[1]}_{args[0]}"][1],
                self._intersection_data["values"][f"{args[1]}_{args[0]}"][0],
            )

    def intersection_index(self, *args: str):
        if {type(arg) for arg in args} != {str}:
            raise TypeError("intersection expects string arguments")
        if len(args) != 2:
            raise ValueError(
                f"intersection takes 2 arguments, {args} provides too many"
            )
        if not hasattr(self, args[0]):
            invalid_arg = args[0]
            raise ValueError(f"SurvivalData does not have {invalid_arg} attribute")
        if not hasattr(self, args[1]):
            invalid_arg = args[1]
            raise ValueError(f"SurvivalData does not have {invalid_arg} attribute")
        if "observed" in args[0] and "censored" in args[1]:
            raise ValueError(
                f"Invalid intersection args {args} : lifetimes can't be observed and censored at the same time"
            )
        if "censored" in args[0] and "observed" in args[1]:
            raise ValueError(
                f"Invalid intersection args {args} : lifetimes can't be observed and censored at the same time"
            )
        if args[0] == "left_censored" and args[1] == "interval_truncated":
            raise ValueError(
                "Left censored lifetimes can't be interval truncated lifetimes too"
            )
        if args[1] == "left_censored" and args[0] == "interval_truncated":
            raise ValueError(
                "Left censored lifetimes can't be interval truncated lifetimes too"
            )
        if args[0] == "left_censored" and args[1] == "left_truncated":
            raise ValueError(
                "Left censored lifetimes can't be left truncated lifetimes too"
            )
        if args[1] == "left_censored" and args[0] == "left_truncated":
            raise ValueError(
                "Left censored lifetimes can't be left truncated lifetimes too"
            )
        if args[0] == args[1]:
            raise ValueError(
                f"Invalid intersection args {args} : args cannot be equals"
            )
        if "censored" in args[0] and "truncated" in args[1]:
            return self._intersection_data["index"][f"{args[0]}_{args[1]}"]
        if "truncated" in args[0] and "censored" in args[1]:
            return (
                self._intersection_data["index"][f"{args[1]}_{args[0]}"][1],
                self._intersection_data["index"][f"{args[1]}_{args[0]}"][0],
            )
        if args[0] == "observed" and "truncated" in args[1]:
            return self._intersection_data["index"][f"{args[0]}_{args[1]}"]
        if "truncated" in args[0] and args[1] == "observed":
            return (
                self._intersection_data["index"][f"{args[1]}_{args[0]}"][1],
                self._intersection_data["index"][f"{args[1]}_{args[0]}"][0],
            )

    def union_values(self, *args):
        if {type(arg) for arg in args} != {str}:
            raise TypeError("union expects string arguments")
        if {hasattr(self, arg) for arg in args} != {True}:
            raise ValueError("some arg are not attribute of SurvData")
        args = set(args)
        return (getattr(self, f"{arg}").values for arg in args)

    def union_index(self, *args):
        if {type(arg) for arg in args} != {str}:
            raise TypeError("union expects string arguments")
        if {hasattr(self, arg) for arg in args} != {True}:
            raise ValueError("some arg are not attribute of SurvData")
        args = set(args)
        return (getattr(self, f"{arg}").index for arg in args)

    def info(self):
        headers = ["Lifetime data", "Counts"]
        info = []
        info.append(["Observed", len(self.observed.values)])
        info.append(["Left censored", len(self.left_censored.values)])
        info.append(["Right censored", len(self.right_censored.values)])
        info.append(["Interval censored", len(self.interval_censored.values)])
        info.append(["Left truncated", len(self.left_truncated.values)])
        info.append(["Right truncated", len(self.right_truncated.values)])
        info.append(["Interval truncated", len(self.interval_truncated.values)])

        row_format = "{:>18}" * (len(headers))
        print(row_format.format(*headers))
        for row in info:
            print(row_format.format(*row))


# factory
def survdata(
    censored_lifetimes: np.ndarray,
    observed_indicators: np.ndarray = None,
    left_censored_indicators: np.ndarray = None,
    right_censored_indicators: np.ndarray = None,
    entry: np.ndarray = None,
    departure: np.ndarray = None,
    **kwargs,
) -> SurvivalData:

    observed = kwargs.get("observed", None)
    left_censored = kwargs.get("left_censored", None)
    right_censored = kwargs.get("right_censored", None)
    interval_censored = kwargs.get("interval_censored", None)
    right_truncated = kwargs.get("left_truncated", None)
    left_truncated = kwargs.get("right_truncated", None)
    interval_truncated = kwargs.get("interval_truncated", None)

    if left_censored is None:
        left_censored = left_censored_factory(
            censored_lifetimes, indicators=left_censored_indicators
        )
    else:
        if left_censored_indicators is not None:
            raise ValueError(
                "left_censored_indicators and left_censored can not be specified at the same time"
            )
        if not issubclass(left_censored, Data):
            raise TypeError(f"Data expected, got '{type(left_censored).__name__}'")
    if right_censored is None:
        right_censored = right_censored_factory(
            censored_lifetimes, indicators=right_censored_indicators
        )
    else:
        if right_censored_indicators is not None:
            raise ValueError(
                "right_censored_indicators and right_censored can not be specified at the same time"
            )
        if not issubclass(right_censored, Data):
            raise TypeError(f"Data expected, got '{type(right_censored).__name__}'")

    if left_censored_indicators is not None and right_censored_indicators is not None:
        if len(left_censored_indicators) != len(right_censored_indicators):
            raise ValueError(
                "Expected left_censored_indicators and right_censored_indicators to have the same length"
            )

        if (
            np.stack([left_censored_indicators, right_censored_indicators]).any(axis=0)
            is True
        ):
            raise ValueError(
                "left_censored_indicators and right_censored_indicators can't be true at the same index"
            )

    if interval_censored is None:
        interval_censored = interval_censored_factory(censored_lifetimes)
    else:
        if not issubclass(interval_censored, IntervalData):
            raise TypeError(
                f"IntervalData expected, got '{type(interval_censored).__name__}'"
            )

    if observed_indicators is None:
        if left_censored_indicators:
            observed_indicators = np.stack([not left_censored_indicators])
            if right_censored_indicators:
                observed_indicators = np.stack(
                    [observed_indicators, not right_censored_indicators]
                )
            observed_indicators = observed_indicators.all(axis=0)

    if observed is None:
        observed = observed_factory(censored_lifetimes, indicators=observed_indicators)
    else:
        if observed_indicators is not None:
            raise ValueError(
                "observed_indicators and observed can not be specified at the same time"
            )
        if not issubclass(observed, Data):
            raise TypeError(f"Data expected, got '{type(observed).__name__}'")

    if entry is not None:
        if len(entry) != len(censored_lifetimes):
            raise ValueError(
                "entry values (left truncation values) and censored_lifetimes don't have the same length"
            )
    if departure is not None:
        if len(departure) != len(censored_lifetimes):
            raise ValueError(
                "departure values (right truncation values) and censored_lifetimes don't have the same length"
            )

    if left_truncated is None:
        left_truncated = left_truncated_factory(entry)
    else:
        if entry is not None:
            raise ValueError(
                "entry and left_truncated can not be specified at the same time"
            )
        if not issubclass(left_truncated, Data):
            raise TypeError(f"Data expected, got '{type(left_truncated).__name__}'")
    if right_truncated is None:
        right_truncated = right_truncated_factory(departure)
    else:
        if departure is not None:
            raise ValueError(
                "departure and right_truncated can not be specified at the same time"
            )
        if not issubclass(right_truncated, Data):
            raise TypeError(f"Data expected, got '{type(right_truncated).__name__}'")
    if interval_truncated is None:
        interval_truncated = interval_truncated_factory(
            entry,
            departure,
        )
    else:
        if not issubclass(interval_truncated, IntervalData):
            raise TypeError(
                f"IntervalData expected, got '{type(interval_truncated).__name__}'"
            )
    return SurvivalData(
        observed,
        left_censored,
        right_censored,
        interval_censored,
        left_truncated,
        right_truncated,
        interval_truncated,
    )
