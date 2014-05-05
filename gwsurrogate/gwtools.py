""" Gravitational Wave Surrogate classes for text and hdf5 files"""

from __future__ import division

__copyright__ = "Copyright (C) 2014 GW surrogate group"
__email__     = "sfield@umd.edu"
__status__    = "testing"
__author__    = "Scott Field, Chad Galley"

__license__ = """
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see U{http://www.gnu.org/licenses/}.
"""

import numpy as np


def get_amp_phase(h):
    """ return A(t) and phi(t) where h = A(t) * exp[-i * phi(t)]"""
    # TODO: should check for h's data type (numpy, spline, etc...)

    amp   = np.abs(h)
    phase = 1.0j*np.log(h/amp)
    phase = phase.real
    phase = np.unwrap(phase)

    return amp,phase

def ecc_estimator(time,h,fit_deg,fit_window):
    """" estimate the eccentricity of gravitational wave h 
         from eq 17 of arxiv:1004.4697 (gr-qc)."""

    ### fit phase with degree fit_deg polynomial on fit_window ###
    amp,phase = get_amp_phase(h)
    p_coeff   = np.polyfit(time,phase,fit_deg)
    phase_fit = np.polyval(p_coeff,time)

    ### compute the estimator ###
    ecc_est = ( phase - phase_fit ) / 4

    #return time, ecc_est
    print 'returning phase fit'
    return time, phase_fit

