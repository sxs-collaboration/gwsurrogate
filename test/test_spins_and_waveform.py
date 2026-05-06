"""
Lightweight end-to-end regression test for NRSur7dq4.

Evaluates the surrogate at a fixed parameter point and compares a handful
of mode values against hard-coded expected values.  Catches any regression
introduced by changes to the spline, Wigner-D, or dynamics code paths.

Requires the NRSur7dq4 HDF5 file (~512 MB).  The test fails if the file is
not present.
"""

import os
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Hard-coded expected values
# Generated with the fixed feature-branch code:
#   q=2, chiA0=[0,0,0.5], chiB0=[0,0,-0.3], f_low=0 (native time grid),
#   evaluated at t ≈ -100 M.
# ---------------------------------------------------------------------------
EXPECTED = {
    (2, -2): -7.824097058920128e-02 + 1.938088511764885e-01j,
    (2,  0): -2.057018634775000e-04 + 3.022657009148731e-06j,
    (2,  2): -7.824928526991039e-02 - 1.938267559950780e-01j,
    (3, -1):  2.012090684917627e-04 - 5.061592210096143e-05j,
    (3,  3):  1.810621264394530e-03 - 2.116647178186893e-02j,
}
EXPECTED_SUM_ABS = 4.955406483976013e-01

# Tolerances — loose enough for numerical round-trip but tight enough to
# catch real regressions.
RTOL = 1e-6
ATOL = 1e-10


# ---------------------------------------------------------------------------
# Model data check
# ---------------------------------------------------------------------------

def _model_path():
    """Return the path to the NRSur7dq4.h5 file, or None if absent."""
    import gwsurrogate as gws  # noqa: PLC0415
    candidate = os.path.join(
        os.path.dirname(gws.__file__),
        "surrogate_downloads",
        "NRSur7dq4.h5",
    )
    return candidate if os.path.isfile(candidate) else None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nrsur7dq4_result():
    """Load the surrogate once per module and return (t, h)."""
    import gwsurrogate as gws  # noqa: PLC0415
    import warnings

    model_path = _model_path()
    assert model_path is not None, (
        "NRSur7dq4.h5 not found; run test/download_regression_models.py first"
    )

    sur = gws.LoadSurrogate("NRSur7dq4")
    q = 2.0
    chiA0 = np.array([0.0, 0.0, 0.5])
    chiB0 = np.array([0.0, 0.0, -0.3])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t, h, _ = sur(q=q, chiA0=chiA0, chiB0=chiB0, f_low=0.0)

    # Evaluation index closest to t = -100 M
    idx = int(np.argmin(np.abs(t + 100.0)))
    return t, h, idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_nan_in_waveform(nrsur7dq4_result):
    """The waveform must be NaN-free at all times."""
    _, h, _ = nrsur7dq4_result
    for mode, values in h.items():
        assert not np.any(np.isnan(values)), f"NaN detected in mode {mode}"


def test_mode_values_at_t_minus100(nrsur7dq4_result):
    """Mode values at t ≈ -100 M match hard-coded expected values."""
    _, h, idx = nrsur7dq4_result
    for mode, expected in EXPECTED.items():
        val = h[mode][idx]
        np.testing.assert_allclose(val, expected, rtol=RTOL, atol=ATOL,
                                   err_msg=f"Mode {mode} regressed")


def test_total_amplitude_at_t_minus100(nrsur7dq4_result):
    """Sum of |h_lm| across all modes at t ≈ -100 M matches expected."""
    _, h, idx = nrsur7dq4_result
    total = sum(np.abs(h[m][idx]) for m in h)
    np.testing.assert_allclose(
        total, EXPECTED_SUM_ABS, rtol=RTOL, atol=ATOL,
        err_msg="Total mode amplitude regressed",
    )


def test_time_array_properties(nrsur7dq4_result):
    """Time array should be monotonically increasing and contain t=0."""
    t, _, _ = nrsur7dq4_result
    assert np.all(np.diff(t) > 0), "Time array is not monotonically increasing"
    assert t[0] < 0 < t[-1], "Time array should span t=0"
