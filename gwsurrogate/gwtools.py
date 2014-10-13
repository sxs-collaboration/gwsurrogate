"""a collection of useful gravitational wave tools"""

from __future__ import division

__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@astro.cornell.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Scott Field, Chad Galley"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def amp_phase(h):
    """Get amplitude and phase of waveform, h = A*exp(i*phi)"""

    # TODO: check for nans returned by phase computation
    amp = np.abs(h);
    return amp, np.unwrap( np.real( -1j * np.log( h/amp ) ) )


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def find_instant_freq(hp, hc, t):
    """instantaneous starting frequency for 

                h = A(t) exp(2 * pi * i * f(t) * t), 

       where we approximate \partial_t A ~ \partial_t f ~ 0."""

    h    = hp + 1j*hc
    dt   = t[1] - t[0]
    hdot = (h[2] - h[0]) / (2 * dt) # 2nd order derivative approximation at t[1]

    f_instant = hdot / (2 * np.pi * 1j * h[1])
    f_instant = f_instant.real

    return f_instant

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def remove_amplitude_zero(t,h):
    """ removes h[i] t[i] from array if |h[i]| = 0 """

    amp, phase     = amp_phase(h)
    where_non_zero = np.nonzero(amp)

    return t[where_non_zero], h[where_non_zero]


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def modify_phase(h,offset):
    """ Modify GW mode's phase to be \phi(t) -> \phi(t) + offset """

    return  h*np.exp(1.0j * offset)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def coordinate_time_shift(t,offset):
    """ modify times to be t -> t + offset """

    return t + offset


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def find_common_time_window(t1,t2):
    """ given two temporal grids, find the largest range of common times 
        defined by [min_common,max_common] """

    min_common = max( t1[0], t2[0] )
    max_common = min( t1[-1], t2[-1] )
    
    if (max_common <= min_common):
        raise ValueError("there is no common time grid")

    return min_common, max_common

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def simple_align_params(t1,h1,t2,h2):
    """ t1 times for complex waveform h1. 

        This routine returns simple alignment parameters 
        deltaT and deltaPhi by...
          (i)  fining discrete waveform peak
          (ii) aligning phase values at the peak time """

    amp1,phase1 = amp_phase(h1)
    amp2,phase2 = amp_phase(h2)

    deltaT   = t1[np.argmax( amp1 )] - t2[np.argmax( amp2 )]
    deltaPhi = phase1[np.argmax( amp1 )] - phase2[np.argmax( amp2 )]

    return deltaT, deltaPhi

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def generate_parameterized_norm(t,h1_interp,h2_eval,mynorm):
    """ 
Input: t is an array of times such that h2_eval = h2[t]
       h2_eval is an array of waveform evaluations
       h1_interp is an interpolant of a waveform 
       mynorm is a function s.t. mynorm(f,dt) is a discrete norm

Output: this routine will return a parameterized norm 

           || h1(deltaT,deltaPhi) - h2_eval || 

which can be minimized over (deltaT,deltaPhi). If deltaT = deltaPhi = 0 then 
the norm is exactly 

                   || h1_eval - h2_eval ||   for h1_eval = h1_interp[t]

Input expecations: h1_interp should be defined on a larger temporal grid 
than t and, hence, h2_eval.  When looking for the minimum h1_interp
will be evaluated at times t + deltaT. t should be viewed as the 
"common set of times" on which both h1 and h2 are known. """

    dt = 1.0 # we optimize the relative errors, factors of dt cancel for Euclidean norm
    
    def ParameterizedNorm(x):

        deltaT_off   = x[0]
        deltaPhi_off = x[1]
        
        t_offset = coordinate_time_shift(t,-deltaT_off)
        h1_eval  = h1_interp(t_offset)
        h1_trial = modify_phase(h1_eval,-deltaPhi_off)

        err_h          = h1_trial - h2_eval
        overlap_errors = mynorm(err_h,dt)/mynorm(h1_trial,dt)
    
        return overlap_errors
    
    return ParameterizedNorm

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def minimize_norm_error(t1,h1,t2,h2,mynorm,t_low_adj=.1,t_up_adj=.1,method='nelder-mead'):

    t1, h1 = remove_amplitude_zero(t1,h1)
    t2, h2 = remove_amplitude_zero(t2,h2)

    deltaT, deltaPhi = simple_align_params(t1,h1,t2,h2)
    h1               = modify_phase(h1,-deltaPhi)
    t1               = coordinate_time_shift(t1,-deltaT)

    h1_interp = interp1d(t1,h1)
    h2_interp = interp1d(t2,h2)

    common_dt      = (t1[2] - t1[1])
    t_start, t_end = find_common_time_window(t1,t2)
    common_times   = np.arange(t_start+t_low_adj,t_end-t_up_adj,common_dt) # small buffer needed 

    h1_eval = h1_interp(common_times)
    h2_eval = h2_interp(common_times)

    ParameterizedNorm = generate_parameterized_norm(common_times,h1_interp,h2_eval,mynorm)
    print "Preliminary guess: %e" %(ParameterizedNorm([0.0,0.0]))

    if method == 'nelder-mead':
        print "optimization with nelder-mead..."
        res_nm = minimize(ParameterizedNorm, [0.0,0.0], method='nelder-mead',tol=1e-12,options={'disp': True})

        print "(NM) the global minimizer is  %e , %e " %(res_nm.x[0],res_nm.x[1])
        print "(NM) for which the objective functional is %e " %ParameterizedNorm([ res_nm.x[0], res_nm.x[1]])
    else:
        raise ValueError("not a valid minimization method")

