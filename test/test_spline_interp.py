"""
Unit tests for gwsurrogate.spline_interp_Cwrapper.

Verifies `interpolate` and `interpolate_many` against scipy CubicSpline
(natural BCs) and cross-validates the batch function against the loop.
"""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from gwsurrogate.spline_interp_Cwrapper import interpolate, interpolate_many, interpolate_many_complex


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _natural_scipy(x, y, xnew):
    """Reference: scipy CubicSpline with natural boundary conditions."""
    return CubicSpline(x, y, bc_type="natural")(xnew)


# ---------------------------------------------------------------------------
# interpolate — single-row tests
# ---------------------------------------------------------------------------

def test_interpolate_polynomial():
    """Linear functions are reproduced exactly (y''=0 is consistent with natural BCs)."""
    x = np.linspace(-2.0, 2.0, 20)
    y = 3.5 * x - 1.2           # linear: y'' = 0, satisfies natural BC exactly
    xnew = np.linspace(-1.9, 1.9, 200)
    result = interpolate(xnew, x, y)
    expected = 3.5 * xnew - 1.2
    np.testing.assert_allclose(result, expected, atol=1e-10,
                               err_msg="linear polynomial not reproduced")


def test_interpolate_matches_scipy():
    """interpolate agrees with scipy CubicSpline (natural BCs) on a smooth function."""
    x = np.sort(RNG.uniform(0.0, 5.0, 30))
    y = np.sin(x) + 0.5 * np.cos(3.0 * x)
    xnew = np.linspace(x[0], x[-1], 300)
    result = interpolate(xnew, x, y)
    expected = _natural_scipy(x, y, xnew)
    np.testing.assert_allclose(result, expected, atol=1e-10,
                               err_msg="interpolate does not match scipy CubicSpline")


def test_interpolate_endpoint_values():
    """Interpolated values at the knot points should match exactly."""
    x = np.linspace(0.0, 1.0, 15)
    y = np.exp(x)
    result = interpolate(x, x, y)
    np.testing.assert_allclose(result, y, atol=1e-12,
                               err_msg="interpolate does not reproduce knot values")


def test_interpolate_extrapolation_raises():
    """Asking for a point outside [x[0], x[-1]] must raise."""
    x = np.linspace(0.0, 1.0, 10)
    y = np.sin(x)
    with pytest.raises(Exception):
        interpolate(np.array([1.1]), x, y)
    with pytest.raises(Exception):
        interpolate(np.array([-0.1]), x, y)


def test_interpolate_complex_via_split():
    """Complex interpolation via split real/imag matches direct use."""
    x = np.linspace(0.0, 2 * np.pi, 25)
    y = np.exp(1j * x) + 0.3 * np.exp(-2j * x)
    xnew = np.linspace(0.1, 6.0, 100)

    re = interpolate(xnew, x, np.real(y))
    im = interpolate(xnew, x, np.imag(y))
    combined = re + 1j * im

    ref_re = _natural_scipy(x, np.real(y), xnew)
    ref_im = _natural_scipy(x, np.imag(y), xnew)
    np.testing.assert_allclose(combined.real, ref_re, atol=1e-10)
    np.testing.assert_allclose(combined.imag, ref_im, atol=1e-10)


# ---------------------------------------------------------------------------
# interpolate_many — batch tests
# ---------------------------------------------------------------------------

def test_interpolate_many_matches_loop():
    """interpolate_many agrees with a row-by-row loop of interpolate."""
    M, N = 8, 40
    x = np.linspace(0.0, 4.0, N)
    y = RNG.standard_normal((M, N))
    xnew = np.linspace(0.2, 3.8, 120)

    result_many = interpolate_many(xnew, x, y)
    result_loop = np.array([interpolate(xnew, x, y[i]) for i in range(M)])

    np.testing.assert_allclose(result_many, result_loop, rtol=1e-12,
                               err_msg="interpolate_many disagrees with loop of interpolate")


def test_interpolate_many_shape():
    """Output shape is (M, len(xnew))."""
    M, N = 5, 30
    x = np.linspace(0.0, 1.0, N)
    y = RNG.standard_normal((M, N))
    xnew = np.linspace(0.0, 1.0, 50)
    result = interpolate_many(xnew, x, y)
    assert result.shape == (M, len(xnew)), (
        f"Expected shape {(M, len(xnew))}, got {result.shape}"
    )


def test_interpolate_many_single_row():
    """interpolate_many with one row matches interpolate."""
    x = np.linspace(0.0, 3.0, 25)
    y = np.sin(x)
    xnew = np.linspace(0.1, 2.9, 80)

    result_many = interpolate_many(xnew, x, y[np.newaxis, :])
    result_single = interpolate(xnew, x, y)

    np.testing.assert_allclose(result_many[0], result_single, rtol=1e-12,
                               err_msg="interpolate_many single row disagrees with interpolate")


def test_interpolate_many_dtype():
    """Output dtype should be float64."""
    x = np.linspace(0.0, 1.0, 20)
    y = RNG.standard_normal((4, 20)).astype(np.float32)   # intentional float32 input
    xnew = np.linspace(0.1, 0.9, 30)
    result = interpolate_many(xnew, x, y)
    assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"


def test_interpolate_many_noncontiguous_input():
    """interpolate_many handles non-C-contiguous (e.g. transposed) y arrays."""
    N = 50
    M = 4
    x = np.linspace(0.0, 5.0, N)
    # Create a Fortran-ordered / non-contiguous y via transpose
    y_base = RNG.standard_normal((N, M))  # shape (N, M), C-contiguous
    y = y_base.T                          # shape (M, N), NOT C-contiguous
    assert not y.flags["C_CONTIGUOUS"], "test setup: y should be non-contiguous"

    xnew = np.linspace(0.2, 4.8, 120)

    result = interpolate_many(xnew, x, y)
    result_loop = np.array([interpolate(xnew, x, y[i]) for i in range(M)])

    np.testing.assert_allclose(result, result_loop, rtol=1e-12,
                               err_msg="interpolate_many fails on non-contiguous input")


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorBounds:
    """Error code 2: evaluation point outside data range."""

    def test_interpolate_above(self):
        x = np.linspace(0.0, 1.0, 10)
        y = np.sin(x)
        with pytest.raises(RuntimeError, match="outside data range"):
            interpolate(np.array([1.5]), x, y)

    def test_interpolate_below(self):
        x = np.linspace(0.0, 1.0, 10)
        y = np.sin(x)
        with pytest.raises(RuntimeError, match="outside data range"):
            interpolate(np.array([-0.5]), x, y)

    def test_interpolate_many_above(self):
        x = np.linspace(0.0, 1.0, 10)
        y = np.sin(x)[np.newaxis, :]
        with pytest.raises(RuntimeError, match="outside data range"):
            interpolate_many(np.array([1.5]), x, y)

    def test_interpolate_many_complex_above(self):
        x = np.linspace(0.0, 1.0, 10)
        y = (np.sin(x) + 1j * np.cos(x))[np.newaxis, :]
        with pytest.raises(RuntimeError, match="outside data range"):
            interpolate_many_complex(np.array([1.5]), x, y)


class TestErrorSingular:
    """Error code 3: zero-length interval (duplicate knots)."""

    def test_interpolate_duplicate_knots(self):
        x = np.array([0.0, 0.5, 0.5, 1.0])  # duplicate at 0.5
        y = np.array([0.0, 1.0, 1.0, 0.0])
        with pytest.raises(RuntimeError, match="Zero-length interval"):
            interpolate(np.array([0.25]), x, y)

    def test_interpolate_many_duplicate_knots(self):
        x = np.array([0.0, 0.5, 0.5, 1.0])
        y = np.array([[0.0, 1.0, 1.0, 0.0]])
        with pytest.raises(RuntimeError, match="Zero-length interval"):
            interpolate_many(np.array([0.25]), x, y)

    def test_interpolate_many_complex_duplicate_knots(self):
        x = np.array([0.0, 0.5, 0.5, 1.0])
        y = np.array([[0.0+0j, 1.0+1j, 1.0+1j, 0.0+0j]])
        with pytest.raises(RuntimeError, match="Zero-length interval"):
            interpolate_many_complex(np.array([0.25]), x, y)


class TestErrorDataSize:
    """Error code 4: data_size < 3."""

    def test_interpolate_two_points(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        with pytest.raises(RuntimeError, match="data_size must be >= 3"):
            interpolate(np.array([0.5]), x, y)

    def test_interpolate_many_two_points(self):
        x = np.array([0.0, 1.0])
        y = np.array([[0.0, 1.0]])
        with pytest.raises(RuntimeError, match="data_size must be >= 3"):
            interpolate_many(np.array([0.5]), x, y)

    def test_interpolate_many_complex_two_points(self):
        x = np.array([0.0, 1.0])
        y = np.array([[0.0+0j, 1.0+1j]])
        with pytest.raises(RuntimeError, match="data_size must be >= 3"):
            interpolate_many_complex(np.array([0.5]), x, y)


class TestErrorOutSize:
    """Error code 5: out_size <= 0."""

    def test_interpolate_empty_xnew(self):
        x = np.linspace(0.0, 1.0, 10)
        y = np.sin(x)
        with pytest.raises(RuntimeError, match="out_size must be > 0"):
            interpolate(np.array([]), x, y)

    def test_interpolate_many_empty_xnew(self):
        x = np.linspace(0.0, 1.0, 10)
        y = np.sin(x)[np.newaxis, :]
        with pytest.raises(RuntimeError, match="out_size must be > 0"):
            interpolate_many(np.array([]), x, y)

    def test_interpolate_many_complex_empty_xnew(self):
        x = np.linspace(0.0, 1.0, 10)
        y = (np.sin(x) + 1j * np.cos(x))[np.newaxis, :]
        with pytest.raises(RuntimeError, match="out_size must be > 0"):
            interpolate_many_complex(np.array([]), x, y)


class TestErrorNumDatasets:
    """Error code 6: num_datasets <= 0."""

    def test_interpolate_many_zero_datasets(self):
        x = np.linspace(0.0, 1.0, 10)
        y = np.empty((0, 10), dtype=np.float64)
        with pytest.raises(RuntimeError, match="num_datasets must be > 0"):
            interpolate_many(np.array([0.5]), x, y)

    def test_interpolate_many_complex_zero_datasets(self):
        x = np.linspace(0.0, 1.0, 10)
        y = np.empty((0, 10), dtype=np.complex128)
        with pytest.raises(RuntimeError, match="num_datasets must be > 0"):
            interpolate_many_complex(np.array([0.5]), x, y)


# ---------------------------------------------------------------------------
# Accuracy tests (continued)
# ---------------------------------------------------------------------------

def test_interpolate_many_matches_scipy():
    """Each row of interpolate_many matches scipy CubicSpline (natural BCs)."""
    M, N = 6, 35
    x = np.linspace(0.0, np.pi, N)
    y = np.stack([np.sin((i + 1) * x) for i in range(M)])
    xnew = np.linspace(0.05, np.pi - 0.05, 200)

    result = interpolate_many(xnew, x, y)
    for i in range(M):
        expected = _natural_scipy(x, y[i], xnew)
        np.testing.assert_allclose(result[i], expected, atol=1e-10,
                                   err_msg=f"Row {i} disagrees with scipy")


# ---------------------------------------------------------------------------
# interpolate_many_complex — complex batch tests
# ---------------------------------------------------------------------------

def test_interpolate_many_complex_matches_split():
    """interpolate_many_complex matches split real/imag via interpolate_many."""
    M, N = 6, 40
    x = np.linspace(0.0, 4.0, N)
    y_complex = RNG.standard_normal((M, N)) + 1j * RNG.standard_normal((M, N))
    xnew = np.linspace(0.2, 3.8, 120)

    result = interpolate_many_complex(xnew, x, y_complex)

    re = interpolate_many(xnew, x, np.real(y_complex))
    im = interpolate_many(xnew, x, np.imag(y_complex))
    expected = re + 1j * im

    np.testing.assert_allclose(result.real, expected.real, atol=1e-12,
                               err_msg="Real part mismatch")
    np.testing.assert_allclose(result.imag, expected.imag, atol=1e-12,
                               err_msg="Imag part mismatch")


def test_interpolate_many_complex_matches_scipy():
    """Each row matches scipy CubicSpline on real and imag separately."""
    M, N = 4, 30
    x = np.linspace(0.0, 2 * np.pi, N)
    y_complex = np.stack([np.exp(1j * (k + 1) * x) for k in range(M)])
    xnew = np.linspace(0.1, 2 * np.pi - 0.1, 100)

    result = interpolate_many_complex(xnew, x, y_complex)
    for i in range(M):
        ref_re = _natural_scipy(x, np.real(y_complex[i]), xnew)
        ref_im = _natural_scipy(x, np.imag(y_complex[i]), xnew)
        np.testing.assert_allclose(result[i].real, ref_re, atol=1e-10,
                                   err_msg=f"Row {i} real part disagrees with scipy")
        np.testing.assert_allclose(result[i].imag, ref_im, atol=1e-10,
                                   err_msg=f"Row {i} imag part disagrees with scipy")


def test_interpolate_many_complex_shape_dtype():
    """Output shape is (M, len(xnew)), dtype is complex128."""
    M, N = 5, 25
    x = np.linspace(0.0, 1.0, N)
    y = RNG.standard_normal((M, N)) + 1j * RNG.standard_normal((M, N))
    xnew = np.linspace(0.0, 1.0, 50)
    result = interpolate_many_complex(xnew, x, y)
    assert result.shape == (M, len(xnew)), (
        f"Expected shape {(M, len(xnew))}, got {result.shape}"
    )
    assert result.dtype == np.complex128, f"Expected complex128, got {result.dtype}"


def test_interpolate_many_complex_single_row():
    """Single-row complex input matches split real/imag via interpolate."""
    x = np.linspace(0.0, 3.0, 25)
    y = np.sin(x) + 1j * np.cos(x)
    xnew = np.linspace(0.1, 2.9, 80)

    result = interpolate_many_complex(xnew, x, y[np.newaxis, :])
    re = interpolate(xnew, x, np.real(y))
    im = interpolate(xnew, x, np.imag(y))

    np.testing.assert_allclose(result[0].real, re, atol=1e-12,
                               err_msg="Single row real part mismatch")
    np.testing.assert_allclose(result[0].imag, im, atol=1e-12,
                               err_msg="Single row imag part mismatch")
