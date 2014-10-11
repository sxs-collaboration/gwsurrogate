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

