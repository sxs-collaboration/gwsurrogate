"""
Unit tests for rotateWaveform.

Tests:
  - Identity quaternion leaves waveform unchanged
  - Shape and dtype preserved
  - Unitarity: rotating by q then q^{-1} recovers original
  - Analytic phase change under a time-dependent z-axis rotation
"""

import numpy as np
import pytest

from gwsurrogate.new.precessing_surrogate import rotateWaveform, quatInv
from gwsurrogate.precessing_utils import _utils


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
        assert h_rot.dtype == h.dtype


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


class TestRotateWaveformZRotation:
    """A z-axis rotation should apply the known phase to every mode."""

    @pytest.mark.parametrize(
        ("n_modes", "ell_max"), [(5, 2), (12, 3), (21, 4)]
    )
    def test_z_rotation_diagonal(self, n_modes, ell_max):
        """Z-rotation: D-matrix is diagonal, so each mode picks up a phase."""
        N = 30
        rng = np.random.default_rng(55)
        angle = np.linspace(-0.7 * np.pi, 0.6 * np.pi, N)
        quat = np.zeros((4, N))
        quat[0] = np.cos(angle / 2)
        quat[3] = np.sin(angle / 2)

        h = _random_waveform(n_modes, N, rng)
        h_rot = rotateWaveform(quat.copy(), h)

        # Under this convention a z-axis rotation is diagonal, with the
        # (ell, m) mode acquiring exp(-i*m*angle).
        expected = np.empty_like(h)
        offset = 0
        for ell in range(2, ell_max + 1):
            for m in range(-ell, ell + 1):
                idx = offset + m + ell
                expected[idx] = h[idx] * np.exp(-1j * m * angle)
            offset += 2 * ell + 1

        np.testing.assert_allclose(h_rot, expected, rtol=1e-12, atol=1e-12)


class TestRotateWaveformNativeValidation:
    """The native wrapper must reject arrays that are unsafe for raw access."""

    @staticmethod
    def _inputs(n_times=4):
        quat = np.zeros((4, n_times), dtype=np.float64)
        quat[0] = 1.0
        h = np.arange(5 * n_times, dtype=np.complex128).reshape(5, n_times)
        return quat, h, np.empty_like(h)

    @pytest.mark.parametrize("ell_max", [-1, 0, 1, 9])
    def test_invalid_ellmax(self, ell_max):
        quat, _, _ = self._inputs(n_times=1)
        h = np.empty((0, 1), dtype=np.complex128)
        with pytest.raises(ValueError, match="ellMax must be between 2 and 8"):
            _utils.rotate_waveform(quat, h, ell_max, h.copy())

    def test_supported_upper_ellmax(self):
        n_times = 2
        quat = np.zeros((4, n_times), dtype=np.float64)
        quat[0] = 1.0
        h = np.arange(77 * n_times, dtype=np.complex128).reshape(77, n_times)
        out = np.empty_like(h)
        _utils.rotate_waveform(quat, h, 8, out)
        np.testing.assert_allclose(out, h, atol=1e-12)

    def test_read_only_output(self):
        quat, h, out = self._inputs()
        out.setflags(write=False)
        with pytest.raises(ValueError, match="out must be writable"):
            _utils.rotate_waveform(quat, h, 2, out)

    @pytest.mark.parametrize("array_name", ["h", "out"])
    def test_non_native_byte_order(self, array_name):
        quat, h, out = self._inputs()
        swapped_dtype = np.dtype(np.complex128).newbyteorder("S")
        if array_name == "h":
            h = h.astype(swapped_dtype)
        else:
            out = np.empty(out.shape, dtype=swapped_dtype)
        with pytest.raises(TypeError, match="native-byte-order"):
            _utils.rotate_waveform(quat, h, 2, out)

    @pytest.mark.parametrize("array_name", ["h", "out"])
    def test_misaligned_array(self, array_name):
        quat, h, out = self._inputs()
        target = h if array_name == "h" else out
        buffer = bytearray(target.nbytes + 1)
        misaligned = np.ndarray(
            target.shape, dtype=np.complex128, buffer=buffer, offset=1
        )
        assert misaligned.flags.c_contiguous
        assert not misaligned.flags.aligned
        if array_name == "h":
            h = misaligned
        else:
            out = misaligned
        with pytest.raises(TypeError, match="aligned"):
            _utils.rotate_waveform(quat, h, 2, out)
