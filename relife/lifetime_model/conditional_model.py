import numpy as np

from relife.lifetime_model import FrozenParametricLifetimeModel, ParametricLifetimeModel
from relife import get_nb_assets


def reshape_ar_or_a0(name: str, value):
    value = np.asarray(value)  # in shape : (), (m,) or (m, 1)
    if value.ndim > 2 or (value.ndim == 2 and value.shape[-1] != 1):
        raise ValueError(f"Incorrect {name} shape. Got {value.shape}. Expected (), (m,) or (m, 1)")
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    return value  # out shape: () or (m, 1)


class AgeReplacementModel(ParametricLifetimeModel):
    # noinspection PyUnresolvedReferences
    r"""
    Age replacement model.

    Lifetime model where the assets are replaced at age :math:`a_r`. This is equivalent to the model of :math:`\min(X,a_r)` where
    :math:`X` is a baseline lifetime and :math:`a_r` is the age of replacement.

    Parameters
    ----------
    baseline : any parametric lifetime model (frozen lifetime model works)
        The base lifetime model without conditional probabilities

    Attributes
    ----------
    baseline
    nb_params
    params
    params_names
    plot
    """

    def __init__(self, baseline):
        super().__init__()
        self.baseline = baseline

    def sf(self, time, ar, *args):
        """
        The survival function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.sf(time, *args), 0.0)

    def hf(self, time, ar, *args):
        """
        The hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.hf(time, *args), 0.0)

    def cdf(self, time, ar, *args):
        """
        The cumulative density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        ar = reshape_ar_or_a0("ar", ar)
        return super().cdf(time, *(ar, *args))

    def chf(self, time, ar, *args):
        """
        The cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.chf(time, *args), 0.0)

    def isf(self, probability, ar, *args):
        """
        The inverse of the survival function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).
        """
        ar = reshape_ar_or_a0("ar", ar)
        return np.minimum(self.baseline.isf(probability, *args), ar)

    def ichf(
        self,
        cumulative_hazard_rate,
        ar,
        *args,
    ):
        """
        Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or np.ndarray
            Cumulative hazard rate value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given cumulative hazard rate(s).
        """
        ar = reshape_ar_or_a0("ar", ar)
        return np.minimum(self.baseline.ichf(cumulative_hazard_rate, *args), ar)

    def pdf(self, time, ar, *args):
        """
        The probability density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        ar = reshape_ar_or_a0("ar", ar)
        return np.where(time < ar, self.baseline.pdf(time, *args), 0)

    def mrl(self, time, ar, *args):
        """
        The mean residual life function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        ar = reshape_ar_or_a0("ar", ar)
        ub = np.array(np.inf)
        # ar.shape == (m, 1)
        mask = time >= ar  # (m, 1) or (m, n)
        if np.any(mask):
            time, ub = np.broadcast_arrays(time, ub)
            time = np.ma.MaskedArray(time, mask)  # (m, 1) or (m, n)
            ub = np.ma.MaskedArray(ub, mask)  # (m, 1) or (m, n)
        mu = self.ls_integrate(lambda x: x - time, time, ub, ar, *args, deg=10) / self.sf(
            time, ar, *args
        )  # () or (n,) or (m, n)
        np.ma.filled(mu, 0)
        return np.ma.getdata(mu)

    def ppf(self, probability, ar, *args):
        """
        The percent point function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).
        """
        ar = reshape_ar_or_a0("ar", ar)
        return self.isf(1 - probability, ar, *args)

    # def cdf(self, time, ar, *args):
    #     ar = reshape_ar_or_a0("ar", ar)
    #     return np.where(time < ar, self.baseline.cdf(time, *args), 1.0)

    def median(self, ar, *args):
        """
        The median.

        Parameters
        ----------
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        ar = reshape_ar_or_a0("ar", ar)
        return self.ppf(np.array(0.5), ar, *args)

    def rvs(self, size, ar, *args, nb_assets=None, return_event=False, return_entry=False, seed=None):
        """
        Random variable sampling.

        Parameters
        ----------
        size : int
            Size of the generated sample.
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.
        nb_assets : int, optional
            If nb_assets is not None, 2d arrays of samples are generated.
        return_event : bool, default is False
            If True, returns event indicators along with the sample time values.
        return_entry : bool, default is False
            If True, returns corresponding entry values of the sample time values.
        seed : optional int, np.random.BitGenerator, np.random.Generator, np.random.RandomState, default is None
            If int or BitGenerator, seed for random number generator. If np.random.RandomState or np.random.Generator, use as given.

        Returns
        -------
        float, ndarray or tuple of float or ndarray
            The sample values. If either ``return_event`` or ``return_entry`` is True, returns a tuple containing
            the time values followed by event values, entry values or both.

        Notes
        -----
        If ``return_entry`` is true, returned time values are not residual time. Otherwise, the times are residuals
        """
        ar = reshape_ar_or_a0("ar", ar)
        if nb_assets is None:
            nb_assets = get_nb_assets(ar, *args)
            if nb_assets == 1:
                nb_assets = None
        baseline_rvs = self.baseline.rvs(
            size,
            *args,
            nb_assets=nb_assets,
            return_event=return_event,
            return_entry=return_entry,
            seed=seed,
        )
        time = baseline_rvs[0] if isinstance(baseline_rvs, tuple) else baseline_rvs
        time = np.minimum(time, ar)  # it may change time shape by broadcasting
        if not return_event and not return_entry:
            return time
        elif return_event and not return_entry:
            event = np.broadcast_to(baseline_rvs[1], time.shape).copy()
            event = np.where(time != ar, event, ~event)
            return time, event
        elif not return_event and return_entry:
            entry = np.broadcast_to(baseline_rvs[1], time.shape).copy()
            return time, entry
        else:
            event, entry = baseline_rvs[1:]
            event = np.broadcast_to(event, time.shape).copy()
            entry = np.broadcast_to(entry, time.shape).copy()
            event = np.where(time != ar, event, ~event)
            return time, event, entry

    def ls_integrate(self, func, a, b, ar, *args, deg=10):
        """
        Lebesgue-Stieltjes integration.

        Parameters
        ----------
        func : callable (in : 1 ndarray , out : 1 ndarray)
            The callable must have only one ndarray object as argument and one ndarray object as output
        a : ndarray (maximum number of dimension is 2)
            Lower bound(s) of integration.
        b : ndarray (maximum number of dimension is 2)
            Upper bound(s) of integration. If lower bound(s) is infinite, use np.inf as value.)
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.
        deg : int, default 10
            Degree of the polynomials interpolation

        Returns
        -------
        np.ndarray
            Lebesgue-Stieltjes integral of func from `a` to `b`.
        """
        ar = reshape_ar_or_a0("ar", ar)
        b = np.minimum(ar, b)
        integration = self.baseline.ls_integrate(func, a, b, *args, deg=deg)
        return integration + np.where(b == ar, func(ar) * self.baseline.sf(ar, *args), 0)

    def moment(self, n, ar, *args):
        """
        n-th order moment

        Parameters
        ----------
        n : order of the moment, at least 1.
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        ar = reshape_ar_or_a0("ar", ar)
        return self.ls_integrate(
            lambda x: x**n,
            0,
            np.inf,
            ar,
            *args,
            deg=100,
        )

    def mean(self, ar, *args):
        """
        The mean.

        Parameters
        ----------
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        ar = reshape_ar_or_a0("ar", ar)
        return self.moment(1, ar, *args)

    def var(self, ar, *args):
        """
        The variance.

        Parameters
        ----------
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        ar = reshape_ar_or_a0("ar", ar)
        return self.moment(2, ar, *args) - self.moment(1, ar, *args) ** 2

    def freeze(self, ar, *args):
        """
        Freeze age replacement values and other arguments into the object data.

        Parameters
        ----------
        ar : float or np.ndarray
            Age of replacement values. If ndarray, shape can only be (m,)
            as only one age of replacement per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenAgeReplacementModel
        """
        ar = reshape_ar_or_a0("ar", ar)
        return FrozenAgeReplacementModel(self, ar, *args)


class LeftTruncatedModel(ParametricLifetimeModel):
    # noinspection PyUnresolvedReferences
    r"""Left truncated model.

    Lifetime model where the assets have already reached the age :math:`a_0`.

    Parameters
    ----------
    baseline : any parametric lifetime model (frozen lifetime model works)
        The base lifetime model without conditional probabilities
    nb_params
    params
    params_names
    plot

    Attributes
    ----------
    baseline
    nb_params
    params
    params_names
    plot
    """

    def __init__(self, baseline):
        super().__init__()
        self.baseline = baseline

    def sf(self, time, a0, *args):
        """
        The survival function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return super().sf(time, a0, *args)

    def pdf(self, time, a0, *args):
        """
        The probability density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return super().pdf(time, a0, *args)

    def isf(self, probability, a0, *args):
        """
        The inverse of the survival function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).
        """
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        a0 = reshape_ar_or_a0("a0", a0)
        return self.ichf(cumulative_hazard_rate, a0, *args)

    def chf(self, time, a0, *args):
        """
        The cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return self.baseline.chf(a0 + time, *args) - self.baseline.chf(a0, *args)

    def cdf(self, time, a0, *args):
        """
        The cumulative density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return super().cdf(time, *(a0, *args))

    def hf(self, time, a0, *args):
        """
        The hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return self.baseline.hf(a0 + time, *args)

    def ichf(self, cumulative_hazard_rate, a0, *args):
        """
        Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or np.ndarray
            Cumulative hazard rate value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given cumulative hazard rate(s).
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return self.baseline.ichf(cumulative_hazard_rate + self.baseline.chf(a0, *args), *args) - a0

    def rvs(self, size, a0, *args, nb_assets=None, return_event=False, return_entry=False, seed=None):
        """
        Random variable sampling.

        Parameters
        ----------
        size : int
            Size of the generated sample.
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.
        nb_assets : int, optional
            If nb_assets is not None, 2d arrays of samples are generated.
        return_event : bool, default is False
            If True, returns event indicators along with the sample time values.
        return_entry : bool, default is False
            If True, returns corresponding entry values of the sample time values.
        seed : optional int, np.random.BitGenerator, np.random.Generator, np.random.RandomState, default is None
            If int or BitGenerator, seed for random number generator. If np.random.RandomState or np.random.Generator, use as given.

        Returns
        -------
        float, ndarray or tuple of float or ndarray
            The sample values. If either ``return_event`` or ``return_entry`` is True, returns a tuple containing
            the time values followed by event values, entry values or both.
        """
        a0 = reshape_ar_or_a0("a0", a0)
        if nb_assets is None:
            nb_assets = get_nb_assets(a0, *args)
            if nb_assets == 1:
                nb_assets = None
        super_rvs = super().rvs(
            size,
            *(a0, *args),
            nb_assets=nb_assets,
            return_event=return_event,
            return_entry=return_entry,
            seed=seed,
        )
        time = super_rvs[0] if isinstance(super_rvs, tuple) else super_rvs
        complete_ages = time + a0
        output = [time,] # at least time in output
        if return_event:
            event = super_rvs[1] # event always at index 1
            # reconstruct event for AgeReplacementModel c omposition as super skips this info
            if isinstance(self.baseline, AgeReplacementModel):
                ar = reshape_ar_or_a0("ar", args[0])
                event = np.where(complete_ages < ar, event, ~event)
            if isinstance(self.baseline, FrozenAgeReplacementModel):
                ar = reshape_ar_or_a0("ar", self.baseline.args[0])
                event = np.where(complete_ages < ar, event, ~event)
            output.append(event)
        if return_entry:
            output[0] = complete_ages # don't return residual ages
            entry = super_rvs[-1] # entry always at last index
            entry = np.broadcast_to(a0, entry.shape).copy()
            output.append(entry)
        if len(output) > 1:
            return tuple(output) # return tuple, not list
        return output[0]

    def ls_integrate(self, func, a, b, a0, *args, deg=10):
        """
        Lebesgue-Stieltjes integration.

        Parameters
        ----------
        func : callable (in : 1 ndarray , out : 1 ndarray)
            The callable must have only one ndarray object as argument and one ndarray object as output
        a : ndarray (maximum number of dimension is 2)
            Lower bound(s) of integration.
        b : ndarray (maximum number of dimension is 2)
            Upper bound(s) of integration. If lower bound(s) is infinite, use np.inf as value.)
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.
        deg : int, default 10
            Degree of the polynomials interpolation

        Returns
        -------
        np.ndarray
            Lebesgue-Stieltjes integral of func from `a` to `b`.
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return super().ls_integrate(func, a, b, *(a0, *args), deg=deg)

    def mean(self, a0, *args):
        """
        The mean.

        Parameters
        ----------
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return super().mean(*(a0, *args))

    def median(self, a0, *args):
        """
        The median.

        Parameters
        ----------
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return super().median(*(a0, *args))

    def var(self, a0, *args):
        """
        The variance.

        Parameters
        ----------
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return super().var(*(a0, *args))

    def moment(self, n: int, a0, *args):
        """
        n-th order moment

        Parameters
        ----------
        n : int
            Order of the moment, at least 1
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return super().moment(n, *(a0, *args))

    def mrl(self, time, a0, *args):
        """
        The mean residual life function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return super().mrl(time, *(a0, *args))

    def ppf(self, probability, a0, *args):
        """
        The percent point function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).

        Notes
        -----
        The ``ppf`` is the inverse of :py:meth:`~LeftTruncatedModel.cdf`.

        """
        a0 = reshape_ar_or_a0("a0", a0)
        return super().ppf(probability, *(a0, *args))

    def freeze(self, a0, *args):
        """
        Freeze conditional age values and other arguments into the object data.

        Parameters
        ----------
        a0 : float or np.ndarray
            Conditional age values. It represents ages reached by assets. If ndarray, shape can only be (m,)
            as only one age per asset can be given
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenLeftTruncatedModel
        """
        a0 = reshape_ar_or_a0("a0", a0)
        return FrozenLeftTruncatedModel(self, a0, *args)


class FrozenAgeReplacementModel(FrozenParametricLifetimeModel):
    r"""
    Frozen age replacement model.

    Parameters
    ----------
    model : AgeReplacementModel
        Any age replacement model.
    ar : float or np.ndarray
        Age of replacement values to be frozen.
    *args : float or np.ndarray
        Additional arguments needed by the model to be frozen.

    Attributes
    ----------
    unfrozen_model : AgeReplacementModel
        The unfrozen age replacement model.
    args : tuple of float or np.ndarray
        All the frozen arguments given and necessary to compute model functions.
    nb_assets : int
        Number of assets passed in frozen arguments. The data is mainly used to control numpy broadcasting and may not
        interest an user.

    Warnings
    --------
    The recommanded way to instanciate a frozen model is by using``freeze`` factory function.
    """

    def __init__(self, model, ar, *args):
        super().__init__(model, *(ar, *args))

    @property
    def ar(self):
        return self.args[0]

    @ar.setter
    def ar(self, value):
        self.args = (value,) + self.args[1:]


class FrozenLeftTruncatedModel(FrozenParametricLifetimeModel):
    r"""
    Frozen left truncated model.

    Parameters
    ----------
    model : LeftTruncatedModel
        Any left truncated model.
    a0 : float or np.ndarray
        Conditional age values to be frozen.
    *args : float or np.ndarray
        Additional arguments needed by the model to be frozen.


    Attributes
    ----------
    unfrozen_model : LeftTruncatedModel
        The unfrozen left truncated model.
    args : tuple of float or np.ndarray
        All the frozen arguments given and necessary to compute model functions.
    nb_assets : int
        Number of assets passed in frozen arguments. The data is mainly used to control numpy broadcasting and may not
        interest an user.

    Warnings
    --------
    The recommanded way to instanciate a frozen model is by using``freeze`` factory function.
    """

    def __init__(self, model, a0, *args):
        super().__init__(model, *(a0, *args))

    @property
    def a0(self):
        return self.args[0]

    @a0.setter
    def a0(self, value):
        self.args = (value,) + self.args[1:]
