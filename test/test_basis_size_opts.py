"""Tests for basis-size options.

The option-validation tests use a minimal synthetic HDF5 file and apply to any
domain-decomposed model that supports basis sizing. The final integration test
uses NRSur7dq4v2 to check basis reconstruction against a complete model file.
"""

import gc
import os

import h5py
import numpy as np
import pytest

from gwsurrogate.new.precessing_surrogate import (
    _validate_basis_size_opts,
)


# Model-independent option validation

@pytest.fixture
def component_file(tmp_path):
    """Create a model-agnostic HDF5 component for option validation."""
    path = tmp_path / "basis-options.h5"
    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("hCoorb_2_0_real_subdomain_0")
        group.create_dataset("nodeIndices", data=np.arange(3))
        yield h5file


def test_valid_basis_size_dictionary(component_file):
    """Any compatible model accepts integer sizes for known datapieces."""
    opts = _validate_basis_size_opts(
        component_file, {"2_0_real_sd_0": np.int64(2)}
    )
    assert opts == {"2_0_real_sd_0": 2}


@pytest.mark.parametrize("opts", ["Fast", [], 2, 1.5, object()])
def test_invalid_basis_size_opts_type(component_file, opts):
    """Unsupported top-level option types fail before model construction."""
    with pytest.raises(TypeError, match="basis_size_opts must be"):
        _validate_basis_size_opts(component_file, opts)


def test_unknown_basis_size_key(component_file):
    """Datapiece typos cannot be silently ignored by any supported model."""
    with pytest.raises(ValueError, match="not_a_datapiece"):
        _validate_basis_size_opts(
            component_file, {"not_a_datapiece": 1}
        )


@pytest.mark.parametrize(
    ("basis_size", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (4, ValueError),
        (1.5, TypeError),
        (True, TypeError),
    ],
)
def test_invalid_basis_size_value(component_file, basis_size, error_type):
    """Invalid counts cannot create empty, unintended, or oversized slices."""
    with pytest.raises(error_type, match="Basis size"):
        _validate_basis_size_opts(
            component_file, {"2_0_real_sd_0": basis_size}
        )


# NRSur7dq4v2 integration coverage

def _nrsur7dq4v2_path():
    import gwsurrogate as gws

    return os.path.join(
        os.path.dirname(gws.__file__),
        "surrogate_downloads",
        "NRSur7dq4v2.h5",
    )


def test_full_size_dictionary_reproduces_nrsur7dq4v2(monkeypatch):
    """A custom full-size preset reproduces every NRSur7dq4v2 coorbital mode."""
    import gwsurrogate as gws
    from gwsurrogate.new import _model_presets

    model_path = _nrsur7dq4v2_path()
    assert os.path.isfile(model_path), (
        "NRSur7dq4v2.h5 not found; run "
        "test/download_regression_models.py first"
    )

    full_surrogate = gws.LoadSurrogate("NRSur7dq4v2")
    full_coorbital = full_surrogate._sur_dimless.coorb_sur
    full_basis_sizes = full_surrogate.coorbital_basis_sizes(ellMax=5)

    chi_a = np.zeros((len(full_coorbital.t), 3))
    chi_b = np.zeros_like(chi_a)
    full_modes = full_coorbital(2.0, chi_a, chi_b, ellMax=5)
    full_times = full_coorbital.t.copy()

    del full_coorbital
    del full_surrogate
    gc.collect()

    custom_preset_name = "TestFullBasisSizes"
    monkeypatch.setitem(
        _model_presets.MODEL_PRESETS["NRSur7dq4v2"], custom_preset_name, full_basis_sizes
    )
    restricted_surrogate = gws.LoadSurrogate("NRSur7dq4v2", model_preset=custom_preset_name)
    restricted_coorbital = restricted_surrogate._sur_dimless.coorb_sur
    restricted_modes = restricted_coorbital(2.0, chi_a, chi_b, ellMax=5)

    np.testing.assert_array_equal(restricted_coorbital.t, full_times)
    np.testing.assert_allclose(
        restricted_modes, full_modes, rtol=2e-11, atol=1e-12
    )
