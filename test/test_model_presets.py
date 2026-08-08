"""Model-independent tests for model-scoped evaluation presets."""

import pytest

from gwsurrogate import surrogate as surrogate_module
from gwsurrogate.new import _basis_presets, _model_presets


def test_resolve_model_preset_is_model_scoped(monkeypatch):
    """The same preset name may have different meanings for different models."""
    monkeypatch.setitem(
        _model_presets.MODEL_PRESETS,
        "DifferentModel",
        {"Fast": {"different_datapiece": 3}},
    )

    nr_options = _model_presets.resolve_model_preset("NRSur7dq4v2", "Fast")
    other_options = _model_presets.resolve_model_preset(
        "DifferentModel", "Fast"
    )
    assert nr_options == _basis_presets.Fast
    assert other_options == {"different_datapiece": 3}


def test_resolve_unknown_model_preset():
    """Unknown model and preset combinations produce a clear error."""
    with pytest.raises(ValueError, match="Unknown model_preset"):
        _model_presets.resolve_model_preset("DifferentModel", "Fast")


def test_resolve_model_preset_returns_independent_copy():
    """Callers cannot accidentally mutate the registered preset."""
    options = _model_presets.resolve_model_preset("NRSur7dq4v2", "Fast")
    options["2_0_real_sd_0"] = 1

    fresh_options = _model_presets.resolve_model_preset(
        "NRSur7dq4v2", "Fast"
    )
    assert fresh_options["2_0_real_sd_0"] != 1


def test_resolve_model_preset_rejects_non_string_name():
    """Preset names have one stable, documented type."""
    with pytest.raises(TypeError, match="model_preset must be a string"):
        _model_presets.resolve_model_preset("NRSur7dq4v2", 1)


def test_load_surrogate_resolves_model_preset(tmp_path, monkeypatch):
    """The generic loader resolves the preset registered for its model."""
    class ModelWithBasisPreset:
        def __init__(self, filename, basis_tol_opts=None):
            self.filename = filename
            self.basis_tol_opts = basis_tol_opts

    model_name = "ModelWithBasisPreset"
    model_path = tmp_path / (model_name + ".h5")
    model_path.touch()
    monkeypatch.setitem(
        surrogate_module.SURROGATE_CLASSES, model_name, ModelWithBasisPreset
    )
    monkeypatch.setitem(
        _model_presets.MODEL_PRESETS,
        model_name,
        {"Fast": {"datapiece": 3}},
    )
    monkeypatch.setattr(
        surrogate_module,
        "SURROGATES_WITH_BASIS_SIZE_OPTS",
        [model_name],
    )

    model = surrogate_module.LoadSurrogate(
        str(model_path), model_preset="Fast"
    )
    assert model.basis_tol_opts == {"datapiece": 3}


def test_load_surrogate_rejects_preset_with_custom_basis_sizes(
        tmp_path, monkeypatch):
    """Preset guarantees cannot be mixed with unvalidated custom overrides."""
    class FutureModel:
        pass

    model_name = "FutureModel"
    model_path = tmp_path / (model_name + ".h5")
    model_path.touch()
    monkeypatch.setitem(
        surrogate_module.SURROGATE_CLASSES, model_name, FutureModel
    )

    with pytest.raises(ValueError, match="either model_preset or"):
        surrogate_module.LoadSurrogate(
            str(model_path),
            model_preset="Fast",
            basis_size_opts={"datapiece": 1},
        )
