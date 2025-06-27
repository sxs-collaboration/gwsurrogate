import numpy as np

from .interp_utils import spline_interpolation_many

from time import time

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
    debug=False,
):
    kwargs = {
        "approximant": approximant,
        "ell_max": ellMax,
        "Omega_1": omega_start,
        "Omega_e": omega_end,
    }
    if dt is not None:
        # If dt is specified, instead of passing it directly we first ask the PN
        # code to be interpolated onto a time grid with a fixed number of saves
        # per orbit. Passing dt directly is too slow, as a dense time grid gets
        # created before the transformation to the inertial frame, which would
        # be very expensive. Instead, we get good enough sampling with
        # saves_per_orbit, and after we get the waveform in the inertial frame,
        # we can interpolate it to the desired dt. saves_per_orbit should be
        # high enough to ensure small interpolation errors in the latter step.
        # We generally want it to be at least 2*ellMax (which would be the
        # Nyquist frequency for the highest-m mode).
        kwargs["saves_per_orbit"] = max(
            2 * ellMax, 10
        )  # At least 10 saves per orbit

    if drop_memory_terms:
        # if we want to drop m=0 modes, we ask for the waveform in the
        # PN coorbital frame.
        kwargs["inertial"] = False

    if debug:
        start_pnevolve = time()

    # Evolve PN
    w = PNWaveform(
        q / (1 + q),
        1 / (1 + q),
        np.array(chiA0),
        np.array(chiB0),
        omega_ref,
        **kwargs,
    )

    if debug:
        end_pnevolve = time()
        print(
            f"PN evolution took {(end_pnevolve - start_pnevolve) * 1000:.2f} ms"
        )

    # Get the dynamics
    # At this point t=0 corresponds to the reference frequency omega_ref.
    t = w.t
    chiA = w.chi1
    chiB = w.chi2
    # quat = None
    # omega_orb = w.v**3 / (w.M1 + w.M2)
    # phi_orb = w.orbital_phase

    if debug:
        start_transform = time()

    if drop_memory_terms:
        # We are in the PN coorbital frame, so set all m=0 modes to zero
        for ell in range(2, ellMax + 1):
            m0_idx = w.index(ell, 0)
            w.data.T[m0_idx] *= 0
        # Now go to the inertial frame
        w = w.to_inertial_frame()

    if debug:
        end_transform = time()
        print(
            f"Transform to inertial frame took {(end_transform - start_transform) * 1000:.2f} ms"
        )

    # We are in the inertial frame. We get the coprecessing frame
    # for use in the hybridization.
    # w_copr = w.to_coprecessing_frame()
    # quat_copr = w_copr.frame

    if debug:
        start_interp = time()

    if dt is not None:
        # Now we interpolate the waveform to the desired dt if needed.

        # Time array including t=0 so that the reference point is a data point
        # So the start and end times must be integer multiples of dt.
        # Also ensure no extrapolation.
        t_new = np.arange(
            int(np.ceil(t[0] / dt)) * dt,
            int(np.floor(t[-1] / dt)) * dt + dt,
            dt,
        )
        # Drop the first point if it falls before t_[0], similarly for the last point.
        if t_new[0] < t[0]:
            t_new = t_new[1:]
        if t_new[-1] > t[-1]:
            t_new = t_new[:-1]

        chiA = spline_interpolation_many(t_new, t, chiA.T).T
        chiB = spline_interpolation_many(t_new, t, chiB.T).T
        h = spline_interpolation_many(t_new, t, w.data.T)
    else:
        # If dt is None, we just use the original time array and data
        t_new = t
        h = w.data.T

    if debug:
        end_interp = time()
        print(f"Interpolation took {(end_interp - start_interp) * 1000:.2f} ms")

    # Convert to dict format
    h_dict = {}
    mode_idx = 0
    for ell in range(2, ellMax + 1):
        for m in range(-ell, ell + 1):
            h_dict[(ell, m)] = h[mode_idx]
            mode_idx += 1

    t_new += t_ref  # Sets t=t_ref at omega_ref

    return t_new, h_dict, chiA, chiB  # , omega_orb, phi_orb, quat_copr


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
