from __future__ import annotations

import itertools
from collections.abc import Generator
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import pytest
from optype.numpy import ArrayND

if TYPE_CHECKING:
    from abc import ABC

    # https://stackoverflow.com/questions/65334215/how-can-request-param-be-annotated-in-indirect-parametrization
    from pytest import FixtureRequest as _FixtureRequest

    T = TypeVar("T")

    class FixtureRequest(_FixtureRequest, ABC, Generic[T]):
        param: T
else:
    pass


def _generate_shapes(
    n: int, num_axes: int
) -> Generator[tuple[tuple[int, ...], ...]] | Generator[tuple[int, ...]]:
    """
    Generate sets of mutually broadcastable shapes.

    Parameters
    ----------
    n : int
        The number of shapes to generate in each set.
    num_axes : int
        The number of axes

    Examples
    --------
    >>> list(_generate_shapes(1, 2))
    [(), (1, 1), (1, 4), (2, 1), (2, 4)]
    >>> list(_generate_shapes(2, 2))
    [((), ()),
     ((), (1, 1)),
     ((), (1, 4)),
     ((), (2, 1)),
     ((), (2, 4)),
     ((1, 1), ()),
     ((1, 1), (1, 1)),
     ((1, 1), (1, 4)),
     ((1, 1), (2, 1)),
     ((1, 1), (2, 4)),
     ((1, 4), ()),
     ((1, 4), (1, 1)),
     ((1, 4), (1, 4)),
     ((1, 4), (2, 1)),
     ((1, 4), (2, 4)),
     ((2, 1), ()),
     ((2, 1), (1, 1)),
     ((2, 1), (1, 4)),
     ((2, 1), (2, 1)),
     ((2, 1), (2, 4)),
     ((2, 4), ()),
     ((2, 4), (1, 1)),
     ((2, 4), (1, 4)),
     ((2, 4), (2, 1)),
     ((2, 4), (2, 4))]
    """
    shape_patterns = itertools.product([0, 1], repeat=num_axes)
    shape_ref = np.arange(2, 2 * num_axes + 1, 2)
    ones_ref = np.ones_like(shape_ref)
    shapes: list[tuple[int, ...]] = []
    for pattern in shape_patterns:
        shape = np.where(pattern, shape_ref, ones_ref)
        if np.all(shape == 1):
            shapes.append(())
        shapes.append(tuple(shape.tolist()))
    if n > 1:
        yield from itertools.product(shapes, repeat=n)
    else:
        yield from shapes


@pytest.fixture(
    params=list(_generate_shapes(1, 4)),
    ids=lambda shape: f"time:{shape}",
)
def time(request: FixtureRequest[tuple[int, ...]]) -> ArrayND[np.float64]:
    return np.ones(request.param, dtype=np.float64)


@pytest.fixture(
    params=list(_generate_shapes(1, 4)),
    ids=lambda shape: f"probability:{shape}",
)
def probability(request: FixtureRequest[tuple[int, ...]]) -> ArrayND[np.float64]:
    return 0.5 * np.ones(request.param, dtype=np.float64)


@pytest.fixture(
    params=list(_generate_shapes(1, 4)),
    ids=lambda shape: f"size:{shape}",
)
def rvs_size(request: FixtureRequest[tuple[int, ...]]) -> tuple[int, ...]:
    return request.param


@pytest.fixture(
    params=list(_generate_shapes(3, 2)),
    ids=lambda shape: f"time:{shape[0]} - covars:{shape[1:]}",
)
def time_covar(
    request: FixtureRequest[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
) -> tuple[ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]]:
    time_shape, covar_shape_0, covar_shape_1 = request.param
    return (
        np.ones(time_shape, dtype=np.float64),
        np.ones(covar_shape_0, dtype=np.float64),
        np.ones(covar_shape_1, dtype=np.float64),
    )


@pytest.fixture(
    params=list(_generate_shapes(3, 2)),
    ids=lambda shape: f"size:{shape}",
)
def rvs_size_covar(
    request: FixtureRequest[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
) -> tuple[tuple[int, ...], ArrayND[np.float64], ArrayND[np.float64]]:
    rvs_size, covar_shape_0, covar_shape_1 = request.param
    return (
        rvs_size,
        np.ones(covar_shape_0, dtype=np.float64),
        np.ones(covar_shape_1, dtype=np.float64),
    )


@pytest.fixture(
    params=list(_generate_shapes(3, 2)),
    ids=lambda shape: f"probability:{shape[0]} - covars:{shape[1:]}",
)
def probability_covar(
    request: FixtureRequest[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
) -> tuple[ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]]:
    probability_shape, covar_shape_0, covar_shape_1 = request.param
    return (
        0.5 * np.ones(probability_shape, dtype=np.float64),
        np.ones(covar_shape_0, dtype=np.float64),
        np.ones(covar_shape_1, dtype=np.float64),
    )


@pytest.fixture(
    params=list(_generate_shapes(4, 2)),
    ids=lambda shape: f"a:{shape[0]} - b:{shape[1]} - covars:{shape[2:]}",
)
def a_b_covar(
    request: FixtureRequest[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    ],
) -> tuple[
    ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64], ArrayND[np.float64]
]:
    a_shape, b_shape, covar_shape_0, covar_shape_1 = request.param
    return (
        np.ones(a_shape, dtype=np.float64),
        np.ones(b_shape, dtype=np.float64),
        np.ones(covar_shape_0, dtype=np.float64),
        np.ones(covar_shape_1, dtype=np.float64),
    )


@pytest.fixture(
    params=list(_generate_shapes(1, 4)),
    ids=lambda shape: f"bound_a:{shape}",
)
def integration_bound_a(
    request: FixtureRequest[tuple[int, ...]],
) -> ArrayND[np.float64]:
    return 2.0 * np.ones(request.param, dtype=np.float64)


@pytest.fixture(
    params=list(_generate_shapes(1, 4)),
    ids=lambda shape: f"bound_b:{shape}",
)
def integration_bound_b(
    request: FixtureRequest[tuple[int, ...]],
) -> ArrayND[np.float64]:
    return 8.0 * np.ones(request.param, dtype=np.float64)


@pytest.fixture(
    params=list(_generate_shapes(1, 4)),
    ids=lambda shape: f"ar:{shape}",
)
def ar(request: FixtureRequest[tuple[int, ...]]) -> ArrayND[np.float64]:
    return np.ones(request.param, dtype=np.float64)
