# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import pytest
import numpy as np

from relife.data import LifetimeData

# def test_data_format_1D():
#     """
#     Test the formatting of 1D time data in LifetimeData class.
#     """
#     time = np.array([1,2,4,5,5,7,10,10,2,10,10,11])
#     event = np.array([0,0,0,1,1,0,0,1,0,0,1,0])
#     assert len(time) == len(event), "Time and event arrays must have the same length"
#     l = LifetimeData(time)
#     assert False 


def test_data_format_2D():
    """
    Test the formatting of 2D time data in LifetimeData class.
    Validates the shapes and values of time, event, entry, and interval censoring attributes.
    """
    time2d = np.array([[1,2],[0,4],[5,5],[7,np.inf],[10,10], [2,10], [10,11]])
    l = LifetimeData(time2d)

    expected_event = np.array([0, 0, 1, 0, 1, 0, 0])
    assert np.isclose(l.event, expected_event).all(), "Event values do not match expected"
    
    expected_xl = np.array([1., 0., 5., 7., 10., 2., 10.])
    assert np.isclose(l.xl, expected_xl).all(), "xl values do not match expected"

    expected_xr = np.array([2., 4., 5., np.inf, 10., 10., 11.])
    assert np.isclose(l.xr, expected_xr).all(), "xr values do not match expected"

    # index of values in function of event
    expected_D = np.array([[5.], [10.]])
    assert np.isclose(l._time.D.flatten(), expected_D.flatten()).all(), "D values do not match expected"

    expected_LC = np.array([[4.]])
    assert np.isclose(l._time.LC.flatten(), expected_LC.flatten()).all(), "LC values do not match expected"

    expected_D_RC = np.array([[5.], [7.], [10.]])
    assert np.isclose(l._time.D_RC.flatten(), expected_D_RC.flatten()).all(), "D_RC values do not match expected"

    expected_IC = np.array([[1., 2.], [2., 10.], [10., 11.]])
    assert np.isclose(l._time.IC.flatten(), expected_IC.flatten()).all(), "IC values do not match expected"

    # Validate shapes
    assert l.time.shape == (7, 2), "Unexpected shape for time attribute"
    assert l.entry.shape == (7,), "Unexpected shape for entry attribute"
    assert l.event.shape == (7,), "Unexpected shape for event attribute"
    assert l._time.D.shape == (2, 1), "Unexpected shape for D attribute"
    assert l._time.LC.shape == (1,1), f"Unexpected shape for LC attribute, got {l._time.LC}"