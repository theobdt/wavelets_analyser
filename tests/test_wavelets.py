import numpy as np
from numpy.testing import assert_array_equal
import wavelets
import pytest


@pytest.fixture()
def data_3d():
    data = np.array([[[0.96, 0.61, 0.82],
                      [0.74, 0.32, 0.25],
                      [0.27, 0.77, 0.91],
                      [0.13, 0.02, 0.75]],
                     [[0.27, 0.54, 0.03],
                      [0.30, 0.72, 0.14],
                      [0.62, 0.98, 0.07],
                      [0.00, 0.33, 0.02]],
                     [[0.28, 0.56, 0.09],
                      [0.62, 0.34, 0.44],
                      [0.83, 0.47, 0.06],
                      [0.74, 0.07, 0.81]],
                     [[0.45, 0.44, 0.28],
                      [0.57, 0.36, 0.59],
                      [0.07, 0.97, 0.28],
                      [0.29, 0.96, 0.54]],
                     [[0.28, 0.60, 0.24],
                      [0.97, 0.03, 0.85],
                      [0.60, 0.63, 0.12],
                      [0.58, 0.94, 0.47]]])
    return data


def test_local_maxima_3D(data_3d):
    coordinates, values = wavelets.local_maxima_3D(data_3d, order=1)

    exp_coordinates = np.array([[1, 2, 1], [3, 2, 1]])
    exp_values = np.array([0.98, 0.97])

    assert_array_equal(coordinates, exp_coordinates)
    assert_array_equal(values, exp_values)


def test_filter_coordinates(data_3d):
    coordinates, _ = wavelets.local_maxima_3D(data_3d, order=1)
    filtered = wavelets.filter_coordinates(data_3d, coordinates)
    exp_filtered = np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0.98, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0.97, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]])
    assert_array_equal(filtered, exp_filtered)
