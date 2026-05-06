"""
Unit tests for _wignerD_matrices.

Uses analytic properties of Wigner D matrices:
  - Identity quaternion → identity matrices
  - Unitarity: D @ D†  = I
  - Shape / dtype
  - Keep copy of python implementation as a regression going forward.
"""

import numpy as np
import pytest

from gwsurrogate.new.precessing_surrogate import (
    _wignerD_matrices,
    _assemble_powers,
)
from gwsurrogate.precessing_utils import _utils


# ---------------------------------------------------------------------------
# Python reference implementation
# ---------------------------------------------------------------------------

def _wignerD_matrices_python(q, ellMax):
    """Pure-Python reference: the original implementation before C migration."""
    ra = q[0] + 1.j * q[3]
    rb = q[2] + 1.j * q[1]
    ra_small = (abs(ra) < 1.e-12)
    rb_small = (abs(rb) < 1.e-12)
    i1 = np.where((1 - ra_small) * (1 - rb_small))[0]
    i2 = np.where(ra_small)[0]
    i3 = np.where((1 - ra_small) * rb_small)[0]

    n = len(ra)
    lvals = range(2, ellMax + 1)
    matrices = [0.j * np.zeros((2 * ell + 1, 2 * ell + 1, n)) for ell in lvals]

    for i, ell in enumerate(lvals):
        for m in range(-ell, ell + 1):
            if (ell + m) % 2 == 1:
                matrices[i][ell + m, ell - m, i2] = rb[i2] ** (2 * m)
            else:
                matrices[i][ell + m, ell - m, i2] = -1 * rb[i2] ** (2 * m)
            matrices[i][ell + m, ell + m, i3] = ra[i3] ** (2 * m)

    ra_i1 = ra[i1]
    rb_i1 = rb[i1]
    ra_pows = _assemble_powers(ra_i1, range(-2 * ellMax, 2 * ellMax + 1))
    rb_pows = _assemble_powers(rb_i1, range(-2 * ellMax, 2 * ellMax + 1))
    abs_raSqr_pows = _assemble_powers(abs(ra_i1) ** 2, range(0, 2 * ellMax + 1))
    absRRatioSquared = (abs(rb_i1) / abs(ra_i1)) ** 2
    ratio_pows = _assemble_powers(absRRatioSquared, range(0, 2 * ellMax + 1))

    for i, ell in enumerate(lvals):
        for m in range(-ell, ell + 1):
            for mp in range(-ell, ell + 1):
                factor = _utils.wigner_coef(ell, mp, m)
                factor *= ra_pows[2 * ellMax + m + mp]
                factor *= rb_pows[2 * ellMax + m - mp]
                factor *= abs_raSqr_pows[ell - m]
                rhoMin = max(0, mp - m)
                rhoMax = min(ell + mp, ell - m)
                s = 0.
                for rho in range(rhoMin, rhoMax + 1):
                    c = ((-1) ** rho) * (_utils.binom(ell + mp, rho) *
                                         _utils.binom(ell - mp, ell - rho - m))
                    s += c * ratio_pows[rho]
                matrices[i][ell + m, ell + mp, i1] = factor * s

    return matrices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(7)


def _random_unit_quaternions(n):
    """Return (4, n) array of random unit quaternions."""
    q = RNG.standard_normal((4, n))
    q /= np.linalg.norm(q, axis=0)
    return q


def _identity_quaternion(n=1):
    """Return (4, n) identity quaternion(s)."""
    q = np.zeros((4, n))
    q[0] = 1.0
    return q


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_wignerD_shape():
    """Output is a list of length (ellMax-1) with correct shapes and dtype."""
    ellMax = 4
    N = 7
    q = _random_unit_quaternions(N)
    mats = _wignerD_matrices(q, ellMax)
    assert len(mats) == ellMax - 1, f"Expected {ellMax - 1} matrices, got {len(mats)}"
    for i, ell in enumerate(range(2, ellMax + 1)):
        expected_shape = (2 * ell + 1, 2 * ell + 1, N)
        assert mats[i].shape == expected_shape, (
            f"ell={ell}: expected shape {expected_shape}, got {mats[i].shape}"
        )
        assert mats[i].dtype == np.complex128, (
            f"ell={ell}: expected complex128, got {mats[i].dtype}"
        )


def test_wignerD_identity_quaternion():
    """Identity quaternion → identity matrix for every ell."""
    ellMax = 4
    q = _identity_quaternion(n=1)
    mats = _wignerD_matrices(q, ellMax)
    for i, ell in enumerate(range(2, ellMax + 1)):
        D = mats[i][:, :, 0]
        size = 2 * ell + 1
        np.testing.assert_allclose(
            D, np.eye(size, dtype=complex), rtol=1e-7, atol=1e-12,
            err_msg=f"ell={ell}: D(identity quat) is not the identity matrix",
        )


def test_wignerD_unitarity():
    """D^ell @ D^ell† = I for random unit quaternions."""
    ellMax = 4
    N = 10
    q = _random_unit_quaternions(N)
    mats = _wignerD_matrices(q, ellMax)
    for i, ell in enumerate(range(2, ellMax + 1)):
        size = 2 * ell + 1
        for k in range(N):
            D = mats[i][:, :, k]
            product = D @ D.conj().T
            np.testing.assert_allclose(
                product, np.eye(size, dtype=complex), rtol=1e-7, atol=1e-11,
                err_msg=f"ell={ell}, quat #{k}: D @ D† ≠ I",
            )


def test_wignerD_matches_python_reference():
    """C implementation matches the old Python reference for generic quaternions."""
    ellMax = 4
    N = 8
    # Use only generic quaternions (avoid ra≈0 or rb≈0 edge cases for reference)
    # Construct quaternions where both ra and rb are well away from zero
    q = _random_unit_quaternions(50)
    ra = q[0] + 1j * q[3]
    rb = q[2] + 1j * q[1]
    mask = (np.abs(ra) > 0.1) & (np.abs(rb) > 0.1)
    q_generic = q[:, mask][:, :N]
    assert q_generic.shape[1] == N, "Not enough generic quaternions — adjust seed"

    mats_c = _wignerD_matrices(q_generic, ellMax)
    mats_py = _wignerD_matrices_python(q_generic, ellMax)

    for i, ell in enumerate(range(2, ellMax + 1)):
        np.testing.assert_allclose(
            mats_c[i], mats_py[i], rtol=1e-7, atol=1e-12,
            err_msg=f"ell={ell}: C result disagrees with Python reference",
        )


def test_wignerD_edge_case_ra_small():
    """Quaternions where ra ≈ 0 — C result must still be unitary."""
    ellMax = 3
    # ra = q0 + i*q3 ≈ 0 means q0≈0, q3≈0
    q = np.array([[0.0], [1.0 / np.sqrt(2)], [1.0 / np.sqrt(2)], [0.0]])
    mats = _wignerD_matrices(q, ellMax)
    for i, ell in enumerate(range(2, ellMax + 1)):
        size = 2 * ell + 1
        D = mats[i][:, :, 0]
        np.testing.assert_allclose(
            D @ D.conj().T, np.eye(size, dtype=complex), rtol=1e-7, atol=1e-11,
            err_msg=f"ell={ell}: unitarity fails for ra≈0 quaternion",
        )


def test_wignerD_edge_case_rb_small():
    """Quaternions where rb ≈ 0 — C result must still be unitary."""
    ellMax = 3
    # rb = q2 + i*q1 ≈ 0 means q2≈0, q1≈0
    q = np.array([[1.0 / np.sqrt(2)], [0.0], [0.0], [1.0 / np.sqrt(2)]])
    mats = _wignerD_matrices(q, ellMax)
    for i, ell in enumerate(range(2, ellMax + 1)):
        size = 2 * ell + 1
        D = mats[i][:, :, 0]
        np.testing.assert_allclose(
            D @ D.conj().T, np.eye(size, dtype=complex), rtol=1e-7, atol=1e-11,
            err_msg=f"ell={ell}: unitarity fails for rb≈0 quaternion",
        )


def test_wignerD_multiple_identity_quaternions():
    """Batch of N identity quaternions all give identity matrices."""
    ellMax = 3
    N = 5
    q = _identity_quaternion(n=N)
    mats = _wignerD_matrices(q, ellMax)
    for i, ell in enumerate(range(2, ellMax + 1)):
        size = 2 * ell + 1
        for k in range(N):
            np.testing.assert_allclose(
                mats[i][:, :, k], np.eye(size, dtype=complex), rtol=1e-7, atol=1e-12,
                err_msg=f"ell={ell}, quat #{k}: not identity",
            )


def test_wignerD_invalid_ellmax():
    """ellMax < 2 should raise ValueError."""
    q = _identity_quaternion()
    with pytest.raises(ValueError):
        _wignerD_matrices(q, ellMax=1)


def test_wignerD_invalid_q_shape():
    """q with wrong shape should raise ValueError."""
    with pytest.raises(ValueError):
        _wignerD_matrices(np.ones((3, 5)), ellMax=2)
