"""
Unit tests for rotateWaveform.

Tests:
  - Identity quaternion leaves waveform unchanged
  - Shape and dtype preserved
  - Unitarity: rotating by q then q^{-1} recovers original
  - Consistency between Python batched-matmul and C implementation (once added)
"""

import numpy as np
import pytest

from gwsurrogate.new.precessing_surrogate import rotateWaveform, quatInv


def _random_unit_quaternion(n, rng):
    """Generate n random unit quaternions as (4, n) array."""
    q = rng.standard_normal((4, n))
    q /= np.sqrt(np.sum(q**2, axis=0, keepdims=True))
    return q


def _random_waveform(n_modes, n_times, rng):
    """Generate a random complex waveform array (n_modes, n_times)."""
    return rng.standard_normal((n_modes, n_times)) + 1j * rng.standard_normal(
        (n_modes, n_times)
    )


class TestRotateWaveformIdentity:
    """Rotating by identity quaternion should leave h unchanged."""

    @pytest.mark.parametrize("n_modes", [5, 12, 21])
    def test_identity(self, n_modes):
        rng = np.random.default_rng(42)
        N = 200
        quat = np.zeros((4, N))
        quat[0] = 1.0  # identity quaternion

        h = _random_waveform(n_modes, N, rng)
        h_rot = rotateWaveform(quat.copy(), h)

        np.testing.assert_allclose(h_rot, h, atol=1e-12)


class TestRotateWaveformShape:
    """Output shape and dtype must match input."""

    @pytest.mark.parametrize("n_modes", [5, 12, 21])
    def test_shape_dtype(self, n_modes):
        rng = np.random.default_rng(123)
        N = 50
        quat = _random_unit_quaternion(N, rng)
        h = _random_waveform(n_modes, N, rng)

        h_rot = rotateWaveform(quat.copy(), h)
        assert h_rot.shape == h.shape
        assert h_rot.dtype == np.complex128


class TestRotateWaveformRoundtrip:
    """Rotating by q then by q^{-1} should recover the original waveform."""

    @pytest.mark.parametrize("n_modes", [5, 12, 21])
    def test_roundtrip(self, n_modes):
        rng = np.random.default_rng(999)
        N = 100
        quat = _random_unit_quaternion(N, rng)
        h = _random_waveform(n_modes, N, rng)

        h_rot = rotateWaveform(quat.copy(), h)
        # rotateWaveform internally does quatInv, so to invert the rotation
        # we pass quatInv(quat) — the double inverse gives back original quat
        h_back = rotateWaveform(quatInv(quat.copy()), h_rot)

        np.testing.assert_allclose(h_back, h, atol=1e-10)


class TestRotateWaveformUnitarity:
    """The rotation should preserve the norm of the mode vector at each time step."""

    @pytest.mark.parametrize("n_modes", [5, 12, 21])
    def test_norm_preservation(self, n_modes):
        rng = np.random.default_rng(77)
        N = 150
        quat = _random_unit_quaternion(N, rng)
        h = _random_waveform(n_modes, N, rng)

        h_rot = rotateWaveform(quat.copy(), h)

        # Check norm is preserved per time step
        norm_orig = np.sqrt(np.sum(np.abs(h) ** 2, axis=0))
        norm_rot = np.sqrt(np.sum(np.abs(h_rot) ** 2, axis=0))
        np.testing.assert_allclose(norm_rot, norm_orig, rtol=1e-12)


class TestRotateWaveformConstantQuat:
    """A constant (non-identity) quaternion should give consistent results."""

    def test_z_rotation_diagonal(self):
        """Z-rotation: D-matrix is diagonal, so each mode picks up a phase."""
        N = 30
        rng = np.random.default_rng(55)
        angle = np.pi / 2
        quat = np.zeros((4, N))
        quat[0] = np.cos(angle / 2)
        quat[3] = np.sin(angle / 2)

        h = _random_waveform(5, N, rng)  # ellMax=2
        h_rot = rotateWaveform(quat.copy(), h)

        # Z-rotation → diagonal D-matrix, each mode gets a pure phase.
        # Verify that |h_rot[i]| == |h[i]| per mode (no mixing).
        for idx in range(5):
            np.testing.assert_allclose(
                np.abs(h_rot[idx]), np.abs(h[idx]), atol=1e-12
            )

        # Verify that the phase ratio h_rot/h is constant across time for each mode.
        for idx in range(5):
            ratio = h_rot[idx] / h[idx]
            np.testing.assert_allclose(
                ratio, ratio[0] * np.ones(N), atol=1e-12
            )
