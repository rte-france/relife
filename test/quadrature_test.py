import random

import numpy as np
import pytest
from pytest import approx


from relife.quadrature import legendre_quadrature, laguerre_quadrature

M = 10
N = 3


@pytest.mark.parametrize(
    "a,b",
    [(random.uniform(2.5, 5), random.uniform(8, 10.0))],
)
def test_legendre_0d(a, b):
    integration = legendre_quadrature(np.ones_like, a, b)
    assert integration.shape == ()
    assert integration == approx(b - a)


@pytest.mark.parametrize(
    "a,b",
    [
        (random.uniform(2.5, 5), np.random.uniform(8, 10.0, size=(M,))),
        (np.random.uniform(2.5, 5.0, size=(M,)), random.uniform(8., 10.)),
        (np.random.uniform(2.5, 5.0, size=(M,)), np.random.uniform(8, 10.0, size=(M,))),
    ]
)
def test_legendre_m(a, b):
    integration = legendre_quadrature(np.ones_like, a, b)
    assert integration.shape == (M,)
    assert integration == approx(b - a)


@pytest.mark.parametrize(
    "a,b",
    [
        (random.uniform(2.5, 5), np.random.uniform(8, 10.0, size=(M,N))),
        (np.random.uniform(2.5, 5.0, size=(M, N)), random.uniform(8., 10.)),

        (np.random.uniform(2.5, 5.0, size=(M,1)), np.random.uniform(8, 10.0, size=(1,N))),
        (np.random.uniform(2.5, 5.0, size=(M,1)), np.random.uniform(8, 10.0, size=(M,N))),

        (np.random.uniform(2.5, 5.0, size=(1,N)), np.random.uniform(8, 10.0, size=(M,1))),
        (np.random.uniform(2.5, 5.0, size=(1,N)), np.random.uniform(8, 10.0, size=(M,N))),

        (np.random.uniform(2.5, 5.0, size=(M,N)), np.random.uniform(8, 10.0, size=(M,N))),
    ]
)
def test_legendre_mn(a, b):
    integration = legendre_quadrature(np.ones_like, a, b)
    assert integration.shape == (M, N)
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


