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
    saves_per_orbit_before_interpolation=None,
):
    """
    Generate a post-Newtonian waveform using the Julia PNWaveform package.
    Parameters
    ----------
    q : float
        Mass ratio of the binary (q = m1/m2, where m1 >= m2).
    chiA0 : list or np.ndarray
        Dimensionless spin of the primary BH, specified in the coorbital frame
        at omega_ref.
    chiB0 : list or np.ndarray
        Dimensionless spin of the secondary BH, specified in the coorbital frame
        at omega_ref.
    omega_ref : float
        Reference angular orbital frequency in dimensionless units (rad/M). The
        input spins chiA0 and chiB0 are specified at the point where the PN
        orbital frequency is omega_ref. The returned waveform will be returned
        in an inertial frame that coincides with the coorbital frame at this
        frequency.
    omega_start : float
        Starting angular orbital frequency for the PN evolution in dimensionless
        units (rad/M). Must be less than omega_ref, and the PN evolution will go
        backwards in time until this frequency.
    omega_end : float
        Ending angular orbital frequency for the PN evolution in dimensionless
        units (rad/M). The PN evolution will go forwards in time until this
        frequency.
    t_ref : float
        Reference time in dimensionless units (M). The returned waveform will
        have t=t_ref at omega_ref.
    dt : float or None
        Time step in dimensionless units (M). If specified, the waveform will be
        interpolated to this time step. Note that this dt is not used for dense
        output generation from the ODE solver. That would be too slow as the
        dense array would then need to transform to the inertial frame. Instead,
        we generate a dense output with a fixed number of saves per orbit (see
        the code for the default, but this can be overridden with the
        `saves_per_orbit_before_interpolation` parameter below).
        If dt is None, the waveform will be returned at the time steps used in
        the PN ODE evolution (no dense output).
    approximant : str, optional
        The PN approximant to use. Default is "TaylorT1". Other options are
        "TaylorT4" and "TaylorT5".
    ellMax : int, optional
        Maximum ell to include in the waveform. Default is 5.
    drop_memory_terms : bool, optional
        If True, the m=0 modes will be set to zero, effectively dropping the
        memory terms. This is useful for hybridization with NR waveforms that do
        not include memory terms. Default is True.
    debug : bool, optional
        If True, print debug information about the PN evolution and
        transformation times.
        Default is False.
    saves_per_orbit_before_interpolation : int or None, optional
        If specified, dense output will be generated with this many saves per
        orbit instead of the default in the code. This option is there primarily
        for testing interpolation errors.
    """

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
        # Nyquist frequency for the highest-m mode). But we double that to be
        # safe. In fact, we set it to be at least 20 (4 * ellMax for ellMax=5)
        # so that we have uniformity for most uses cases (ellMax<=5). If it is
        # specified through saves_per_orbit_before_interpolation, we just use
        # that value.
        if saves_per_orbit_before_interpolation is not None:
            kwargs["saves_per_orbit"] = saves_per_orbit_before_interpolation
        else:
            kwargs["saves_per_orbit"] = max(
                4 * ellMax, 20
            )  # At least 20 saves per orbit

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
