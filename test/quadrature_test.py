import random

import numpy as np
import pytest
from pytest import approx


from relife.quadrature import legendre_quadrature, laguerre_quadrature, quadrature

M = 10
N = 3

@pytest.mark.parametrize(
    "a,b",
    [
        (random.uniform(2.5, 5), random.uniform(8, 10.0)),
        (random.uniform(2.5, 5), np.random.uniform(8, 10.0, size=(M,))),
        (random.uniform(2.5, 5), np.random.uniform(8, 10.0, size=(1,N))),
        (random.uniform(2.5, 5), np.random.uniform(8, 10.0, size=(M,1))),
        (random.uniform(2.5, 5), np.random.uniform(8, 10.0, size=(M,N))),

        (np.random.uniform(2.5, 5.0, size=(M,)), random.uniform(8., 10.)),
        (np.random.uniform(2.5, 5.0, size=(M,)), np.random.uniform(8, 10.0, size=(M,))),
        (np.random.uniform(2.5, 5.0, size=(M,)), np.random.uniform(8, 10.0, size=(M,1))),

        (np.random.uniform(2.5, 5.0, size=(M,1)), random.uniform(8., 10.)),
        (np.random.uniform(2.5, 5.0, size=(M,1)), np.random.uniform(8, 10.0, size=(1,N))),
        (np.random.uniform(2.5, 5.0, size=(M,1)), np.random.uniform(8, 10.0, size=(M,N))),

        (np.random.uniform(2.5, 5.0, size=(1,N)), random.uniform(8., 10.)),
        (np.random.uniform(2.5, 5.0, size=(1,N)), np.random.uniform(8, 10.0, size=(M,1))),
        (np.random.uniform(2.5, 5.0, size=(1,N)), np.random.uniform(8, 10.0, size=(M,N))),

        (np.random.uniform(2.5, 5.0, size=(M,N)), random.uniform(8., 10.)),
        (np.random.uniform(2.5, 5.0, size=(M,N)), np.random.uniform(8, 10.0, size=(M,N))),
    ]
)
def test_legendre(a, b):
    integration = legendre_quadrature(np.ones_like, a, b)
    output_shape = np.broadcast_shapes(np.asarray(a).shape, np.asarray(b).shape)
    assert integration.shape == output_shape
    assert integration == approx(b - a)


@pytest.mark.parametrize(
    "a",
    [
        random.uniform(2.5, 5),
        np.random.uniform(2.5, 5.0, size=(M,)),
        np.random.uniform(2.5, 5.0, size=(M,N)),
    ],
)
def test_laguerre(a):
    assert laguerre_quadrature(np.ones_like, a) == approx(np.exp(-a))



@pytest.mark.parametrize(
    "a,b",
    [
        (random.uniform(2.5, 5), random.uniform(8, 10.0)),
        (random.uniform(2.5, 5), np.random.uniform(8, 10.0, size=(M,))),
        (random.uniform(2.5, 5), np.random.uniform(8, 10.0, size=(1,N))),
        (random.uniform(2.5, 5), np.random.uniform(8, 10.0, size=(M,1))),
        (random.uniform(2.5, 5), np.random.uniform(8, 10.0, size=(M,N))),

        (np.random.uniform(2.5, 5.0, size=(M,)), random.uniform(8., 10.)),
        (np.random.uniform(2.5, 5.0, size=(M,)), np.random.uniform(8, 10.0, size=(M,))),
        (np.random.uniform(2.5, 5.0, size=(M,)), np.random.uniform(8, 10.0, size=(M,1))),

        (np.random.uniform(2.5, 5.0, size=(M,1)), random.uniform(8., 10.)),
        (np.random.uniform(2.5, 5.0, size=(M,1)), np.random.uniform(8, 10.0, size=(1,N))),
        (np.random.uniform(2.5, 5.0, size=(M,1)), np.random.uniform(8, 10.0, size=(M,N))),

        (np.random.uniform(2.5, 5.0, size=(1,N)), random.uniform(8., 10.)),
        (np.random.uniform(2.5, 5.0, size=(1,N)), np.random.uniform(8, 10.0, size=(M,1))),
        (np.random.uniform(2.5, 5.0, size=(1,N)), np.random.uniform(8, 10.0, size=(M,N))),

        (np.random.uniform(2.5, 5.0, size=(M,N)), random.uniform(8., 10.)),
        (np.random.uniform(2.5, 5.0, size=(M,N)), np.random.uniform(8, 10.0, size=(M,N))),
    ]
)
def test_quadrature(a, b):
    assert quadrature(np.ones_like, a, b) == approx(b - a)

