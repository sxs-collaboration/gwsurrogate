"""
Unit tests for precessing surrogate utility functions:
  - shared coorbital component evaluation and node counting
  - normalize_spin
  - splinterp_many  (batch spline interpolation)
  - _splinterp_Cwrapper / _splinterp_Cwrapper_many (low-level wrappers)
  - eval_coorb_modes (C vs Python path comparison)
"""

import os
import numpy as np
import pytest

from gwsurrogate.new.precessing_surrogate import (
    _eval_coorbital_component,
    _total_node_evals,
    normalize_spin,
    rotate_spin,
    splinterp_many,
)
from gwsurrogate.new.surrogate import _splinterp_Cwrapper, _splinterp_Cwrapper_many
from gwsurrogate.precessing_utils import _utils


RNG = np.random.default_rng(99)


# ---------------------------------------------------------------------------
# Shared coorbital-surrogate helpers
# ---------------------------------------------------------------------------

def test_eval_coorbital_component_reconstructs_node_fits(monkeypatch):
    """The shared evaluator must preserve node ordering and EI reconstruction."""
    from gwsurrogate.new import precessing_surrogate

    data = {
        'nodeIndices': np.array([1, 0]),
        'orders': [np.array([0]), np.array([1])],
        'coefs': [10., 20.],
        'EI_basis': np.array([[1., 2.], [3., 4.]]),
    }
    chiA = np.array([[1., 2., 3.], [4., 5., 6.]])
    chiB = np.array([[7., 8., 9.], [10., 11., 12.]])
    fit_settings = (0.1, 0.2, 3, 2)
    fit_inputs = []

    def get_fit_params(x):
        fit_inputs.append(x.copy())
        return x

    def eval_fit(orders, coefs, fit_params, *settings):
        assert settings == fit_settings
        return coefs + np.sum(fit_params)

    monkeypatch.setattr(precessing_surrogate._utils, 'eval_fit', eval_fit)
    result = _eval_coorbital_component(
        data, 2., chiA, chiB, get_fit_params, fit_settings
    )

    np.testing.assert_array_equal(result, [216., 328.])
    np.testing.assert_array_equal(fit_inputs[0], [2., 4., 5., 6., 10., 11., 12.])
    np.testing.assert_array_equal(fit_inputs[1], [2., 1., 2., 3., 7., 8., 9.])


def test_total_node_evals_respects_ell_maximum():
    """Shared work accounting must exclude datapieces above the requested ell."""
    data = {
        '2_0_real': {'nodeIndices': np.arange(2)},
        '3_1_Re+': {'nodeIndices': np.arange(1)},
        '4_2_Im-_sd_0': {'nodeIndices': np.arange(3)},
    }

    assert _total_node_evals(data, ellMax=3) == 3
    assert _total_node_evals(data, ellMax=4) == 6


# ---------------------------------------------------------------------------
# Skip condition for tests requiring the NRSur7dq4 model
# ---------------------------------------------------------------------------

def _model_path():
    import gwsurrogate as gws
    candidate = os.path.join(gws.catalog.download_path(), "NRSur7dq4.h5")
    return candidate if os.path.isfile(candidate) else None


_MODEL_AVAILABLE = _model_path() is not None

skip_if_no_model = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="NRSur7dq4.h5 not found",
)


@pytest.fixture(scope="module")
def coorb_test_data():
    """Load NRSur7dq4, run dynamics, return coorb_sur and test inputs."""
    import warnings
    import gwsurrogate as gws

    sur = gws.LoadSurrogate("NRSur7dq4")
    psur = sur._sur_dimless

    q = 2.0
    chiA0 = np.array([0.0, 0.0, 0.5])
    chiB0 = np.array([0.0, 0.0, -0.3])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quat, orbphase, chiA_copr, chiB_copr = \
            psur.get_dynamics(q, chiA0, chiB0)

    from gwsurrogate.new.precessing_surrogate import coorb_spins_from_copr_spins
    from gwsurrogate.new.surrogate import _splinterp_Cwrapper_many

    # Interpolate to coorbital time grid
    t_ds = psur.dynamics_sur.t
    t_coorb = psur.coorb_sur.t
    chiA_copr_coorb = _splinterp_Cwrapper_many(
        t_coorb, t_ds, chiA_copr.T).T
    chiB_copr_coorb = _splinterp_Cwrapper_many(
        t_coorb, t_ds, chiB_copr.T).T
    orbphase_coorb = _splinterp_Cwrapper_many(
        t_coorb, t_ds, orbphase[np.newaxis, :])[0]

    chiA_coorb, chiB_coorb = coorb_spins_from_copr_spins(
        chiA_copr_coorb, chiB_copr_coorb, orbphase_coorb)

    return psur.coorb_sur, q, chiA_coorb, chiB_coorb


@skip_if_no_model
def test_eval_coorb_modes_c_vs_python(coorb_test_data):
    """C eval_coorb_modes matches Python _call_python for NRSur7dq4."""
    coorb_sur, q, chiA, chiB = coorb_test_data
    ellMax = coorb_sur.ellMax  # 4

    modes_c = coorb_sur._call_c(q, chiA, chiB, ellMax)
    modes_py = coorb_sur._call_python(q, chiA, chiB, ellMax)

    np.testing.assert_allclose(modes_c, modes_py, rtol=1e-12, atol=1e-15,
                               err_msg="C and Python coorb modes disagree")


@skip_if_no_model
def test_eval_coorb_modes_partial_ellmax(coorb_test_data):
    """C path with ellMax=2 matches Python path."""
    coorb_sur, q, chiA, chiB = coorb_test_data
    ellMax = 2

    modes_c = coorb_sur._call_c(q, chiA, chiB, ellMax)
    modes_py = coorb_sur._call_python(q, chiA, chiB, ellMax)

    np.testing.assert_allclose(modes_c, modes_py, rtol=1e-12, atol=1e-15,
                               err_msg="Partial ellMax: C vs Python disagree")


@skip_if_no_model
def test_eval_coorb_modes_ellmax_above_model(coorb_test_data):
    """ellMax beyond the model's own ellMax zero-pads the extra modes."""
    coorb_sur, q, chiA, chiB = coorb_test_data
    ellMax = coorb_sur.ellMax + 1

    modes_c = coorb_sur._call_c(q, chiA, chiB, ellMax)
    modes_py = coorb_sur._call_python(q, chiA, chiB, ellMax)

    assert modes_c.shape[0] == ellMax*ellMax + 2*ellMax - 3
    np.testing.assert_allclose(modes_c, modes_py, rtol=1e-12, atol=1e-15,
                               err_msg="ellMax above model: C vs Python differ")


@skip_if_no_model
def test_call_dispatches_to_c(coorb_test_data, monkeypatch):
    """__call__ uses the C path when the fit_params transform is known."""
    coorb_sur, q, chiA, chiB = coorb_test_data
    assert coorb_sur._fit_params_mode >= 0

    def _fail(*args, **kwargs):
        raise AssertionError("__call__ fell back to the Python path")

    monkeypatch.setattr(coorb_sur, "_call_python", _fail)
    modes = coorb_sur(q, chiA, chiB, ellMax=coorb_sur.ellMax)
    assert modes.shape == (coorb_sur.ellMax**2 + 2*coorb_sur.ellMax - 3,
                           len(coorb_sur.t))


@skip_if_no_model
def test_eval_coorb_modes_rejects_bad_arguments(coorb_test_data):
    """Malformed arguments raise ValueError rather than reading out of bounds."""
    coorb_sur, q, chiA, chiB = coorb_test_data
    p = coorb_sur._packed
    ellMax = coorb_sur.ellMax
    nmodes = ellMax*ellMax + 2*ellMax - 3
    q_consts = coorb_sur._compute_q_consts(float(q))
    settings = coorb_sur._fit_settings

    def call(**overrides):
        kw = dict(chiA=chiA, chiB=chiB, nmodes=nmodes, ellMax=ellMax,
                  all_orders=p['all_orders'], q_consts=q_consts)
        kw.update(overrides)
        return _utils.eval_coorb_modes(
            float(q),
            np.ascontiguousarray(kw['chiA'], dtype=np.float64),
            np.ascontiguousarray(kw['chiB'], dtype=np.float64),
            p['comp_n_nodes'], p['comp_node_offset'], p['comp_ell'],
            p['all_node_indices'], p['node_n_coefs'],
            p['node_coef_offset'], p['all_coefs'],
            kw['all_orders'], p['all_EI_basis'],
            p['mode_info'], kw['q_consts'],
            kw['nmodes'], kw['ellMax'], coorb_sur._fit_params_mode,
            settings[0], settings[1], settings[2], settings[3])

    # Baseline: the unmodified arguments are accepted.
    call()

    # chiA too short for the stored node indices.
    with pytest.raises(ValueError):
        call(chiA=chiA[:10], chiB=chiB[:10])

    # nmodes inconsistent with ellMax would let mode_info index past the output.
    with pytest.raises(ValueError):
        call(nmodes=nmodes - 1)

    # Wrong dtype for an array read through a raw typed pointer.
    with pytest.raises(ValueError):
        call(all_orders=p['all_orders'].astype(np.int32))

    # Wrong length for q_consts.
    with pytest.raises(ValueError):
        call(q_consts=q_consts[:3])


# ---------------------------------------------------------------------------
# normalize_spin
# ---------------------------------------------------------------------------

def test_normalize_spin_zero_norm_returns_unchanged():
    """chi_norm == 0 leaves chi unmodified."""
    chi = RNG.standard_normal((10, 3))
    chi_orig = chi.copy()
    normalize_spin(chi, chi_norm=0.0)
    np.testing.assert_array_equal(chi, chi_orig,
                                  err_msg="normalize_spin modified chi when chi_norm=0")


def test_normalize_spin_rescales_rows():
    """Each row of the output has magnitude chi_norm."""
    n = 20
    chi = RNG.standard_normal((n, 3))
    # Make sure no rows are zero
    chi += 0.1
    chi_norm = 0.7

    normalize_spin(chi, chi_norm=chi_norm)
    row_norms = np.sqrt(np.sum(chi**2, axis=1))
    np.testing.assert_allclose(row_norms, chi_norm, rtol=1e-12, atol=1e-14,
                               err_msg="Row magnitudes after normalize_spin are not chi_norm")


def test_normalize_spin_unit_norm():
    """chi_norm=1.0 → all rows become unit vectors."""
    chi = RNG.standard_normal((15, 3)) + 1.0
    normalize_spin(chi, chi_norm=1.0)
    row_norms = np.sqrt(np.sum(chi**2, axis=1))
    np.testing.assert_allclose(row_norms, 1.0, rtol=1e-12, atol=1e-14)


def test_normalize_spin_preserves_direction():
    """normalize_spin only rescales; the direction (unit vector) is unchanged."""
    chi = RNG.standard_normal((12, 3)) + 0.5
    original_chi = chi.copy()
    chi_norm = 0.4
    normalize_spin(chi, chi_norm=chi_norm)
    orig_unit = (original_chi.T / np.sqrt(np.sum(original_chi**2, axis=1))).T
    new_unit = (chi.T / np.sqrt(np.sum(chi**2, axis=1))).T
    np.testing.assert_allclose(new_unit, orig_unit, rtol=1e-12, atol=1e-14,
                               err_msg="normalize_spin changed the direction of chi")


def test_normalize_spin_shape_preserved():
    """In-place normalization preserves the input shape."""
    chi = RNG.standard_normal((8, 3))
    original_shape = chi.shape
    normalize_spin(chi, chi_norm=0.5)
    assert chi.shape == original_shape


# ---------------------------------------------------------------------------
# rotate_spin
# ---------------------------------------------------------------------------

def test_rotate_spin_cached_trig_matches_uncached_path():
    """Precomputed sine and cosine values produce the standard rotation."""
    phase = np.linspace(-np.pi, np.pi, 12)
    chi = RNG.standard_normal((len(phase), 3))

    expected = rotate_spin(chi, phase)
    result = rotate_spin(
        chi, cp=np.cos(phase), sp=np.sin(phase)
    )

    np.testing.assert_allclose(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("provided", ["cp", "sp"])
def test_rotate_spin_requires_cached_trig_pair(provided):
    """Supplying only one cached trigonometric value is an error."""
    phase = np.linspace(0, 1, 4)
    kwargs = {provided: np.ones_like(phase)}

    with pytest.raises(ValueError, match="cp and sp must be provided together"):
        rotate_spin(np.ones((len(phase), 3)), **kwargs)

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

    np.testing.assert_allclose(result_many, result_loop, rtol=1e-12, atol=0,
                               err_msg="splinterp_many disagrees with single-row loop")


def test_splinterp_many_single_row():
    """Single-row splinterp_many matches _splinterp_Cwrapper."""
    t_in = np.linspace(0.0, 2.0, 30)
    t_out = np.linspace(0.1, 1.9, 50)
    y = np.sin(t_in)

    result_many = splinterp_many(t_out, t_in, y[np.newaxis, :])
    result_single = _splinterp_Cwrapper(t_out, t_in, y)

    np.testing.assert_allclose(result_many[0], result_single, rtol=1e-12, atol=0)


def test_splinterp_many_reproduces_knots():
    """Interpolating at the input knots recovers the original data."""
    M, N = 5, 35
    t_in = np.linspace(0.0, 1.0, N)
    data = _make_smooth_rows(M, t_in)

    result = splinterp_many(t_in, t_in, data)
    np.testing.assert_allclose(result, data, rtol=1e-12, atol=1e-12,
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

    np.testing.assert_allclose(result_many, result_loop, rtol=1e-12, atol=0,
                               err_msg="_splinterp_Cwrapper_many disagrees with loop")


def test_splinterp_Cwrapper_many_dtype():
    """Output dtype is float64 for real input."""
    t_in = np.linspace(0.0, 1.0, 20)
    t_out = np.linspace(0.1, 0.9, 30)
    data = np.ones((4, 20), dtype=np.float32)
    result = _splinterp_Cwrapper_many(t_out, t_in, data)
    assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"
