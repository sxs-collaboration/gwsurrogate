from sxs.julia import PNWaveform
from gwsurrogate.new.pn_utils import generate_pn
import lal
import numpy as np
import matplotlib.pyplot as plt

q = 1.1  # Mass ratio
chiA0 = [0.1, 0.2, 0.3]  # Spin of the primary
chiB0 = [0.3, 0.2, 0.1]  # Spin of the secondary
dt_sec = 1 / 4096  # Time step in seconds
f_low_Hz = 20
M = 40  # Total mass in solar masses

MT = lal.MTSUN_SI * M  # Convert total mass to seconds
omega_low = (
    MT * f_low_Hz * np.pi
)  # Convert to angular frequency in dimless units
dt = dt_sec / MT  # Convert from seconds to dimensionless time step

kwargs = {
    "q": q,
    "chiA0": chiA0,
    "chiB0": chiB0,
    "omega_ref": omega_low,
    "omega_start": omega_low * 0.8,
    "omega_end": 0.05,
    "t_ref": 0,
    "dt": dt,
    "ellMax": 5,
}

# With very high saves_per_orbit_before_interpolation
kwargs["saves_per_orbit_before_interpolation"] = 200  # High enough
t, h_dict, chiA, chiB = generate_pn(**kwargs)

# Now try with various saves_per_orbit_before_interpolation
for saves_per_orbit_before_interpolation in [10, 15, 20, 25, 30, 40, 50, 100]:
    kwargs["saves_per_orbit_before_interpolation"] = (
        saves_per_orbit_before_interpolation
    )
    t_2, h_dict_2, chiA_2, chiB_2 = generate_pn(**kwargs)

    # Let's call the longer one t_A and the shorter one t_B
    if len(t) > len(t_2):
        t_A, t_B = t, t_2
        h_dict_A, h_dict_B = h_dict, h_dict_2
    else:
        t_A, t_B = t_2, t
        h_dict_A, h_dict_B = h_dict_2, h_dict

    keep_idx = np.logical_and(t_A >= t_B[0], t_A <= t_B[-1])

    interp_rms_err = 0
    interp_Linf_err = 0
    for key in h_dict.keys():
        # RMS error
        interp_rms_err += np.sqrt(
            np.mean(np.abs(h_dict_A[key][keep_idx] - h_dict_B[key]) ** 2)
        )
        # Linf error
        interp_Linf_err += np.max(
            np.abs(h_dict_A[key][keep_idx] - h_dict_B[key])
        )
    print(
        f"saves_per_orbit={saves_per_orbit_before_interpolation}, "
        f"RMS error: {interp_rms_err:.2e}, "
        f"Linf error: {interp_Linf_err:.2e}"
    )
