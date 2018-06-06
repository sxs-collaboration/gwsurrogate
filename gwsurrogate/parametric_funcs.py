""" catalog of parametric fitting functions """

from __future__ import division # for python 2

__copyright__    = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__        = "sfield@umassd.edu, crgalley@tapir.caltech.edu"
__status__       = "testing"
__author__       = "Jonathan Blackman, Scott Field, Chad Galley"
__contributors__ = []

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
from .gwtools import gwtools as gwtools # from the package gwtools, import the module gwtools (gwtools.py)....



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def polyval_1d(coeffs,x):
  """ 1D polynomial defined by coeffs vector and evaluated 
  as numpy.polyval(coeffs,x)"""
  return np.polyval(coeffs, x)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def ampfitfn1_1d(coeffs,x):
  """ PN inspired ampitude fit a0 + a1*nu**a2"""

  a0, a1, a2 = coeffs[:3]
  nu = gwtools.q_to_nu(x)
  return a0 + a1*nu**a2

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def ampfitfn2_1d(coeffs, x):
  """ PN inspired amplitude fit a0 + a1*np.abs(0.25-nu)**0.5 + a2*np.log(nu/0.25)"""

  a0, a1, a2 = coeffs[:3]
  nu = gwtools.q_to_nu(x)
  return a0 + a1*np.abs(0.25-nu)**0.5 + a2*np.log(nu/0.25)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def phifitfn1_1d(coeffs,x):
  """ PN inspired phase fit a0 + a1*nu + a2*nu**2 + a3*np.log(nu)"""

  a0, a1, a2, a3 = coeffs[:4]
  nu = gwtools.q_to_nu(x)
  return a0 + a1*nu + a2*nu**2 + a3*np.log(nu)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def ampfitfn3_1d(coeffs, x):
  """ PN inspired amplitude fit plus polynomial,
      a0*np.sqrt(1 - x) + a1*np.log(x) + a2(1 - x) + ... + aN*(1-x)**(N-1)"""

  a0 = coeffs[-1]
  a1 = coeffs[-2]

  polyCoefs = [c for c in coeffs[:-2]]
  polyCoefs.append(0.)

  return a0*np.sqrt(1. - x) + a1*np.log(x) + np.polyval(polyCoefs,1. - x)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def ampfitfn4_1d(coeffs, x):
  """ a0*sqrt(1.0-x) + a1*(1.0-x) + a2*(1.0-x)^2 + a3*(1.0-x)^3"""

  a0, a1, a2, a3 = coeffs[:4]
  return a0*np.sqrt(1.0-x) + a1*(1.0-x) + a2*np.power(1.0-x,2) + a3*np.power(1.0-x,3)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def ampfitfn5_1d(coeffs,x):
  """a0*np.sqrt(1 - x) + a1(1 - x) + ... + aN*(1-x)**N"""
  a0 = coeffs[-1]
  polyCoefs = [c for c in coeffs[:-1]]
  polyCoefs.append(0.)
  
  return a0*np.sqrt(1. - x) + np.polyval(polyCoefs,1. - x)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def ampfitfn6_1d(coeffs,x):
  """a0*np.sqrt(1 - x) + a1*(1 - x)**1.5 + [ a2*(1 - x) + ... + aN*(1-x)**(N-1)"""
  a0 = coeffs[-1]
  a1 = coeffs[-2]
  polyCoefs = [c for c in coeffs[:-2]]
  polyCoefs.append(0.)

  return a0*np.sqrt(1. - x) + a1*(1. - x)**1.5 + np.polyval(polyCoefs,1. - x)

### these are for switching from (q,M) to surrogate's parameterization ###
def q_to_q(q):
  """ identity map from q to q
  
  Surrogates with this parameterization expect its user intput 
  to be the mass ratio q, and "map" to the internal surrogate's 
  parameterization which is also q
  
  The surrogates training interval is in mass ratio
  """
  return q

def q_to_nu(q):
  """ map from q to symmetric mass ratio nu
  
  Surrogates with this parameterization expect its user intput 
  to be the mass ratio q. 
  
  The surrogate will map q to the internal surrogate's 
  parameterization which is nu
  
  The surrogates training interval is quoted in symmetric mass ratio.
  """
  return gwtools.q_to_nu(q)

### dictionary of fitting functions ###
function_dict = {
                 "polyval_1d": polyval_1d,
                 "ampfitfn1_1d": ampfitfn1_1d,
                 "ampfitfn2_1d": ampfitfn2_1d,
                 "ampfitfn4_1d": ampfitfn4_1d,
                 "phifitfn1_1d": phifitfn1_1d,
                 "nuSingularPlusPolynomial": ampfitfn5_1d,
                 "nuSingular2TermsPlusPolynomial": ampfitfn6_1d,
                 "q_to_q": q_to_q,
                 "q_to_nu":q_to_nu
                 }
