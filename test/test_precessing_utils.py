"""
Unit tests for precessing surrogate utility functions:
  - normalize_spin
  - splinterp_many  (batch spline interpolation)
  - _splinterp_Cwrapper / _splinterp_Cwrapper_many (low-level wrappers)
"""

import numpy as np
import pytest

from gwsurrogate.new.precessing_surrogate import normalize_spin, splinterp_many
from gwsurrogate.new.surrogate import _splinterp_Cwrapper, _splinterp_Cwrapper_many


RNG = np.random.default_rng(99)


# ---------------------------------------------------------------------------
# normalize_spin
# ---------------------------------------------------------------------------

def test_normalize_spin_zero_norm_returns_unchanged():
    """chi_norm == 0 → chi returned unmodified."""
    chi = RNG.standard_normal((10, 3))
    chi_orig = chi.copy()
    result = normalize_spin(chi, chi_norm=0.0)
    np.testing.assert_array_equal(result, chi_orig,
                                  err_msg="normalize_spin modified chi when chi_norm=0")


def test_normalize_spin_rescales_rows():
    """Each row of the output has magnitude chi_norm."""
    n = 20
    chi = RNG.standard_normal((n, 3))
    # Make sure no rows are zero
    chi += 0.1
    chi_norm = 0.7

    result = normalize_spin(chi, chi_norm=chi_norm)
    row_norms = np.sqrt(np.sum(result**2, axis=1))
    np.testing.assert_allclose(row_norms, chi_norm, rtol=1e-7, atol=1e-14,
                               err_msg="Row magnitudes after normalize_spin are not chi_norm")


def test_normalize_spin_unit_norm():
    """chi_norm=1.0 → all rows become unit vectors."""
    chi = RNG.standard_normal((15, 3)) + 1.0
    result = normalize_spin(chi, chi_norm=1.0)
    row_norms = np.sqrt(np.sum(result**2, axis=1))
    np.testing.assert_allclose(row_norms, 1.0, rtol=1e-7, atol=1e-14)


def test_normalize_spin_preserves_direction():
    """normalize_spin only rescales; the direction (unit vector) is unchanged."""
    chi = RNG.standard_normal((12, 3)) + 0.5
    chi_norm = 0.4
    result = normalize_spin(chi, chi_norm=chi_norm)
    orig_unit = (chi.T / np.sqrt(np.sum(chi**2, axis=1))).T
    new_unit = (result.T / np.sqrt(np.sum(result**2, axis=1))).T
    np.testing.assert_allclose(new_unit, orig_unit, rtol=1e-7, atol=1e-14,
                               err_msg="normalize_spin changed the direction of chi")


def test_normalize_spin_shape_preserved():
    """Output shape equals input shape."""
    chi = RNG.standard_normal((8, 3))
    result = normalize_spin(chi, chi_norm=0.5)
    assert result.shape == chi.shape


# ---------------------------------------------------------------------------
# splinterp_many  (delegates to _splinterp_Cwrapper_many)
# ---------------------------------------------------------------------------

def _make_smooth_rows(M, t_in):
    """M rows of smooth test data sampled at t_in."""
    return np.array([np.sin((i + 1) * t_in) + 0.3 * np.cos(2 * (i + 1) * t_in)
                     for i in range(M)])


def test_splinterp_many_shape():
    """Output shape is (M, len(t_out))."""
    M, N_in, N_out = 7, 50, 80
    t_in = np.linspace(0.0, 5.0, N_in)
    t_out = np.linspace(0.2, 4.8, N_out)
    data = _make_smooth_rows(M, t_in)
    result = splinterp_many(t_out, t_in, data)
    assert result.shape == (M, N_out), f"Expected ({M},{N_out}), got {result.shape}"


def test_splinterp_many_vs_single_loop():
    """splinterp_many agrees with a loop of _splinterp_Cwrapper row by row."""
    M, N_in = 6, 40
    t_in = np.linspace(0.0, np.pi, N_in)
    t_out = np.linspace(0.1, np.pi - 0.1, 100)
    data = _make_smooth_rows(M, t_in)

    result_many = splinterp_many(t_out, t_in, data)
    result_loop = np.array([_splinterp_Cwrapper(t_out, t_in, data[i])
                            for i in range(M)])

    np.testing.assert_allclose(result_many, result_loop, rtol=1e-12,
                               err_msg="splinterp_many disagrees with single-row loop")


def test_splinterp_many_single_row():
    """Single-row splinterp_many matches _splinterp_Cwrapper."""
    t_in = np.linspace(0.0, 2.0, 30)
    t_out = np.linspace(0.1, 1.9, 50)
    y = np.sin(t_in)

    result_many = splinterp_many(t_out, t_in, y[np.newaxis, :])
    result_single = _splinterp_Cwrapper(t_out, t_in, y)

    np.testing.assert_allclose(result_many[0], result_single, rtol=1e-12)


def test_splinterp_many_reproduces_knots():
    """Interpolating at the input knots recovers the original data."""
    M, N = 5, 35
    t_in = np.linspace(0.0, 1.0, N)
    data = _make_smooth_rows(M, t_in)

    result = splinterp_many(t_in, t_in, data)
    np.testing.assert_allclose(result, data, rtol=1e-7, atol=1e-12,
                               err_msg="splinterp_many does not reproduce knot values")


# ---------------------------------------------------------------------------
# _splinterp_Cwrapper_many (low-level)
# ---------------------------------------------------------------------------

def test_splinterp_Cwrapper_many_matches_loop():
    """_splinterp_Cwrapper_many agrees with row-by-row _splinterp_Cwrapper."""
    M, N_in = 9, 45
    t_in = np.linspace(-1.0, 1.0, N_in)
    t_out = np.linspace(-0.9, 0.9, 70)
    data = np.array([np.exp(-(i * 0.5) * t_in**2) for i in range(1, M + 1)])

    result_many = _splinterp_Cwrapper_many(t_out, t_in, data)
    result_loop = np.array([_splinterp_Cwrapper(t_out, t_in, data[i])
                            for i in range(M)])

    np.testing.assert_allclose(result_many, result_loop, rtol=1e-12,
                               err_msg="_splinterp_Cwrapper_many disagrees with loop")


def test_splinterp_Cwrapper_many_dtype():
    """Output dtype is float64 for real input."""
    t_in = np.linspace(0.0, 1.0, 20)
    t_out = np.linspace(0.1, 0.9, 30)
    data = np.ones((4, 20), dtype=np.float32)
    result = _splinterp_Cwrapper_many(t_out, t_in, data)
    assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"
