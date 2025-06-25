import numpy as np

# Start the Julia session and get PNWaveform
from sxs.julia import PNWaveform


def generate_pn(
    q,
    chiA0,
    chiB0,
    omega_ref,
    omega_start,
    omega_end,
    t_ref,
    dt,
    approximant="TaylorT1",  # Use the defaults you actually want as there may be some optimization in the first call
    ellMax=5,
    drop_memory_terms=True,
):

    kwargs = {
        "approximant": approximant,
        "ell_max": ellMax,
        "Omega_1": omega_start,
        "Omega_e": omega_end,
    }
    if dt is not None:
        kwargs["saveat"] = dt

    if drop_memory_terms:
        # if we want to drop m=0 modes, we ask for the waveform in the
        # PN coorbital frame.
        kwargs["inertial"] = False

    # Evolve PN
    w = PNWaveform(
        q / (1 + q),
        1 / (1 + q),
        np.array(chiA0),
        np.array(chiB0),
        omega_ref,
        **kwargs,
    )

    # Get the dynamics
    t = w.t + t_ref  # Sets t=t_ref at omega_ref
    chiA = w.chi1
    chiB = w.chi2
    quat = None
    omega_orb = w.v**3 / (w.M1 + w.M2)
    phi_orb = w.orbital_phase

    if drop_memory_terms:
        # We are in the PN coorbital frame, so set all m=0 modes to zero
        for ell in range(2, ellMax + 1):
            m0_idx = w.index(ell, 0)
            w.data.T[m0_idx] *= 0
        # Now go to the inertial frame
        w = w.to_inertial_frame()

    # We are in the inertial frame. We get the coprecessing frame
    # for use in the hybridization.
    w_copr = w.to_coprecessing_frame()
    quat_copr = w_copr.frame

    # Convert to dict format
    h_dict = {}
    for ell in range(2, ellMax + 1):
        for m in range(-ell, ell + 1):
            h_dict[(ell, m)] = w.data.T[w.index(ell, m)]

    return t, h_dict, chiA, chiB, omega_orb, phi_orb, quat_copr


if __name__ == "__main__":
    # Timing test
    from time import time

    q = 1.1  # Mass ratio
    chiA0 = [0.1, 0.2, 0.3]  # Spin of the primary
    chiB0 = [0.3, 0.2, 0.1]  # Spin of the secondary
    omega_ref = 0.013  # Reference frequency
    omega_start = 0.012379109136936127  # Start frequency of 20Hz for M=40 MSun
    omega_end = 0.027  # End frequency
    t_ref = -1000  # Reference time, can be negative
    dt = 1.2391689667785797  # dt=0.1 converted to seconds for M=40 MSun, so a practical value

    kwargs = {
        "q": q,
        "chiA0": chiA0,
        "chiB0": chiB0,
        "omega_ref": omega_ref,
        "omega_start": omega_start,
        "omega_end": omega_end,
        "t_ref": t_ref,
        "drop_memory_terms": False,  # Let's assume Mike does this in-house eventually
    }

    # Timing with dt specified
    kwargs["dt"] = dt
    # First do a dummy call to make sure everything is loaded and compiled
    generate_pn(**kwargs)
    # Now for the actual timing test
    start_time = time()
    generate_pn(**kwargs)
    end_time = time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms for dt={dt}")

    # Now repeat without specifying dt, which means no need for interpolation
    kwargs["dt"] = None
    # First do a dummy call to make sure everything is loaded and compiled
    generate_pn(**kwargs)
    start_time = time()
    generate_pn(**kwargs)
    end_time = time()
    print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms for dt=None")
