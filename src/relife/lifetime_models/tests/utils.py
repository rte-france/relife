import itertools
from collections.abc import Generator

import numpy as np


def generate_without_duplicates(
    *inputs: list[tuple[int, ...]], repeat: int = 1
) -> Generator[tuple[tuple[int, ...]]]:
    seen = set()
    for prod in itertools.product(*inputs, repeat=repeat):
        prod_set = frozenset(prod)
        if len(prod_set) == 1:  # all items are the same
            continue
        if prod_set not in seen:
            seen.add(prod_set)
            yield prod


def generate_shapes(
    n: int, num_axes: int
) -> list[tuple[tuple[int, ...]]] | list[tuple[int, ...]]:
    """
    Generate sets of mutually broadcastable shapes without repetitions and scalars.

    Parameters
    ----------
    n : int
        The number of shapes to generate in each set.
    num_axes : int
        The number of axes

    Examples
    --------
    >>> _generate_shapes(1, 2)
    [(4,), (2, 1), (2, 4)]
    >>> _generate_shapes(2, 2)
    [((4,), (4,), (4,), (4,), (2, 1)),
     ((4,), (4,), (4,), (4,), (2, 4)),
     ((4,), (4,), (4,), (2, 1), (2, 4)),
     ((2, 1), (2, 1), (2, 1), (2, 1), (2, 4))]
    """
    assert num_axes >= 2
    shape_patterns = itertools.product([0, 1], repeat=num_axes)
    shape_ref = np.arange(2, 2 * num_axes + 1, 2)
    ones_ref = np.ones_like(shape_ref)
    shapes: list[tuple[int, ...]] = []
    for pattern in shape_patterns:
        shape = np.where(pattern, shape_ref, ones_ref)
        mask = np.cumsum(pattern) > 0  # remove first dims if 1
        shape = shape[mask]
        if np.all(shape == 1):  # skip scalar-like
            continue
        shapes.append(tuple(shape.tolist()))
    if n > 1:
        return list(generate_without_duplicates(shapes, repeat=n))
    else:
        return shapes
