""" catalog of parametric fitting functions """

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
import gwtools as gwtools

def polyval_1d(coeffs,x):
    """ 1D polynomial defined by coeffs vector and evaluated 
    as numpy.polyval(coeffs,x)"""
    return np.polyval(coeffs, x)

def ampfitfn_1d(coeffs,x):
    """ PN inspired ampitude fit a0 + a1*nu**a2"""

    a0 = coeffs[0]
    a1 = coeffs[1]
    a2 = coeffs[2]

    nu = gwtools.q_to_nu(x)

    return a0 + a1*nu**a2

def phifitfn_1d(coeffs,x):
    """ PN inspired phase fit a0 + a1*nu + a2*nu**2 + a3*np.log(nu)"""

    a0 = coeffs[0]
    a1 = coeffs[1]
    a2 = coeffs[2]
    a3 = coeffs[3]

    nu = gwtools.q_to_nu(x)

    return a0 + a1*nu + a2*nu**2 + a3*np.log(nu)



### dictionary of fitting functions ###
function_dict = {"polyval_1d":polyval_1d,
                 "ampfitfn_1d":ampfitfn_1d,
                 "phifitfn_1d":phifitfn_1d}
