"""Model-scoped presets for controlling evaluation cost and accuracy.

The registry uses ``MODEL_PRESETS[model_name][preset_name]`` to select a set of
preset options. For NRSur7dq4v2, those options map datapiece names to the number
of basis elements retained. Other models may add their own preset names and
model-specific complexity options in the future.
"""

import copy
from gwsurrogate.new import _basis_presets


MODEL_PRESETS = {
    "NRSur7dq4v2": {
        "Fast": _basis_presets.Fast,
    },
}

# Future models can add their own named presets here. If a future preset
# controls something other than basis size, its model-specific handling can be
# added when that complexity option is introduced.


def resolve_model_preset(model_name, preset_name):
    """Return an independent copy of a named model's preset options."""
    if not isinstance(preset_name, str):
        raise TypeError("model_preset must be a string or None")

    presets = MODEL_PRESETS.get(model_name, {})
    if preset_name not in presets:
        raise ValueError(
            "Unknown model_preset %r for %s. Available presets: %s"
            % (preset_name, model_name, sorted(presets))
        )

    return copy.deepcopy(presets[preset_name])
