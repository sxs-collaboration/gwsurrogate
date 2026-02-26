import numpy as np
import pytest
from gwsurrogate.new.precessing_surrogate import normalize_spin


def test_positive_chi_norm():
    chi = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = normalize_spin(chi, 0.5)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 0.5)


def test_zero_chi_norm():
    chi = np.array([[1.0, 2.0, 3.0]])
    result = normalize_spin(chi, 0.0)
    np.testing.assert_array_equal(result, chi)


def test_negative_chi_norm():
    chi = np.array([[1.0, 2.0, 3.0]])
    result = normalize_spin(chi, -1.0)
    np.testing.assert_array_equal(result, chi)


def test_direction_preserved():
    chi = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    result = normalize_spin(chi, 1.0)
    for i in range(len(chi)):
        expected_dir = chi[i] / np.linalg.norm(chi[i])
        result_dir = result[i] / np.linalg.norm(result[i])
        np.testing.assert_allclose(result_dir, expected_dir)


def test_single_row():
    chi = np.array([[1.0, 0.0, 0.0]])
    result = normalize_spin(chi, 0.8)
    np.testing.assert_allclose(result, [[0.8, 0.0, 0.0]])
