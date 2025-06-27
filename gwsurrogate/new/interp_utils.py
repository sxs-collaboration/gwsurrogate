import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


def spline_interpolation(xout, xin, yin):
    """Interpolating Cubic splines for real or complex data"""

    if np.iscomplexobj(yin):
        re = spline_interpolation(xout, xin, np.real(yin))
        im = spline_interpolation(xout, xin, np.imag(yin))
        return re + 1.0j * im
    else:
        return InterpolatedUnivariateSpline(xin, yin, ext=2)(xout)


def spline_interpolation_many(t_out, t_in, many_things):
    """Interpolating Cubic splines for list of real or complex data"""
    return np.array(
        [spline_interpolation(t_out, t_in, thing) for thing in many_things]
    )
