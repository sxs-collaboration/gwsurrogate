"""
Tensor spline evaluation code.
The intended use case is when we have very many different splines which
all have the same domain (breakpoints). Careful usage of numpy keeps the
added computational cost of using python vs. (for example) C with gsl
minimal when evaluating all splines simultaneously at the same parameter
value.
"""

from __future__ import division  # for py2

__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@astro.cornell.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Jonathan Blackman"

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

#print("__name__ = %s"%__name__)
#print("__package__= %s"%__package__)

if __package__ is "" or "None": # py2 and py3 compatible 
  print("setting __package__ to gwsurrogate.new so relative imports work")
  __package__="gwsurrogate.new"

print("__name__ = %s"%__name__)
print("__package__= %s"%__package__)

import numpy as np
import h5py
from .saveH5Object import SimpleH5Object
import itertools
from functools import reduce

def _cubic_spline_breaks(knot_vec):
    """
Given spline knots [x0, x1, ..., xN-1=xf], we define the breakpoints as
[x0, x0, x0, x0, x1, x2, ..., xN-2, xf, xf, xf, xf]
which has length N+6.
Note that there are N+2 spline coefficients corresponding to bsplines where
coef[i] corresponds to the bspline with support between breakpoints[i] and
breakpoints[i+4], so the supports are:
    i=0:    [x0, x1)
    i=1:    [x0, x2)
    i=2:    [x0, x3)
    i=3:    [x0, x4)
    i=4:    [x1, x5)
    ...
    i=N-3:  [xN-6, xN-2)
    i=N-2:  [xN-5, xf)
    i=N-1:  [xN-4, xf)
    i=N:    [xN-3, xf)
    i=N+1:  [xN-2, xf)
    """

    if np.min(np.diff(knot_vec)) < 0.:
        raise Exception("knot_vec should be strictly increasing")

    return np.append(np.append(np.ones(3)*knot_vec[0], knot_vec),
                     np.ones(3)*knot_vec[-1])

def cubic_spline_breaksToknots(bvec):
    """
Given breakpoints generated from _cubic_spline_breaks,
[x0, x0, x0, x0, x1, x2, ..., xN-2, xf, xf, xf, xf],
return the spline knots [x0, x1, ..., xN-1=xf]. 
This function ``undoes" _cubic_spline_breaks:
knot_vec = _cubic_spline_breaks2knots(_cubic_spline_breaks(knot_vec))
    """

    return bvec[3:-3]


#-----------------------------------------------------------------------------

def _cubic_bspline_eval_nonzero_1d(x, bvec, tol=1.e-12):
    """
Given a point x and an array of breakpoints bvec, determine the 4
potentially non-zero bsplines and evaluate them at x.
Returns i0, bspline_evals.
    """

    knots = bvec[3:-3]

    # Nudge parameters if barely outside domain
    if knots[0] - tol < x <= knots[0]:
        x = knots[0]
    if knots[-1] - tol < x < knots[-1] + tol:
        # The domain is [knots[0], knots[-1]) so nudge inwards a little
        x = knots[-1] - tol

    if x < knots[0] or x > knots[-1]:
        raise Exception("%s outside of [%s, %s] parameter range!"%(
                x, knots[0], knots[-1]))

    if x <= knots[0]:
        i0 = 0
    elif x >= knots[-1]:
        i0 = len(knots) - 2
    else:
        i0 = np.where(x >= knots)[0][-1]

    return i0, np.array([_bspline_eval(x, bvec[i:i+5], k=3)
                         for i in range(i0, i0+4)])

#-----------------------------------------------------------------------------

def _bspline_eval(x, bvec, k, check=True):

    if check and np.min(np.diff(bvec)) < 0.:
        raise Exception("bvec should be strictly increasing")

    if x < bvec[0] or x >= bvec[-1]:
        return 0.

    if k==0:
        return 1.

    c1 = _bspline_eval(x, bvec[:-1], k-1, check=False)
    c2 = _bspline_eval(x, bvec[1:], k-1, check=False)
    if abs(c1) > 0.:
        c1 *= (x - bvec[0])/(bvec[-2] - bvec[0])
    if abs(c2) > 0.:
        c2 *= (bvec[-1] - x)/(bvec[-1] - bvec[1])
    return c1 + c2


def memoize_spline_call(func):
    """Decorator for TensorSplineGrid's __call__ method.

       Multiple calls to __call__ typically have the same xvec
       value, in which case we can return the last result without
       recomputing anything.

       A decorator is needed because member variables of TensorSplineGrid
       are expected to be loaded from an hdf5 file."""

    last_call   = [np.nan]
    last_return = [np.nan, np.nan, np.nan]

    def decorated_function(self,xvec):

        # TODO: make this next code block its own function? Used elsewhere
        xshape = np.shape(xvec)

        # It's convenient to be able to accept a float instead of a length-1
        # array for 1d parameter spaces.

        if len(xshape) == 0:
            xvec = np.array([xvec])
            xshape = np.shape(xvec)

        if np.max(np.abs(last_call[0] - xvec)) != 0:
            last_call[0] = xvec
            last_return[0], last_return[1], last_return[2] = func(self,xvec)
        return last_return[0], last_return[1], last_return[2]
    return decorated_function


class TensorSplineGrid(SimpleH5Object):

    def __init__(self, knot_vecs=[]):
        """
Stores the geometric information needed for tensor-spline interpolation.
No coefficients are stored, so one TensorSplineGrid can be used for many
splines sharing a common grid.
        """

        super(TensorSplineGrid, self).__init__()

        self.dim = len(knot_vecs)
        self.grid_dim = [len(v) for v in knot_vecs]
        self.breakpoint_vecs = [_cubic_spline_breaks(v) for v in knot_vecs]

    def bspline_eval_nonzero(self, xvec):
        """
Returns imin_vals, spline_evals.
Each have length self.dim.
imin_vals gives the minimum spline coefficient index for the spline_evals.
Each element of spline_evals consists of the 4 potentially non-zero
    bspline evaluations.
        """

        res = [_cubic_bspline_eval_nonzero_1d(x, bvec)
               for x, bvec in zip(xvec, self.breakpoint_vecs)]

        imin_vals, spline_evals = [list(t) for t in zip(*res)]
        return imin_vals, spline_evals

    @memoize_spline_call
    def __call__(self, xvec):
        """
Evaluates potentially non-zero spline basis function products.
xvec: The point in parameter space. Must be of type numpy.ndarray
Returns:
    eval_prods: Products of spline evaluations in all parameter space
                directions, which can be summed up with spline coefficients.
    sl: Single slice so we can make better use of np.sum:
    summed_axes: See below

    With these three items, we can evaluate a spline from its coefficients:

       >>> y = np.sum(c[sl] * eval_prods, axis=summed_axes)

    See FastTensorSplineSurrogate's call method

        """

        # All splines use the same grid, so we determine the 4^d potentially
        # non-zero spline basis function products here which can be used for
        # all interpolations.
        imin_vals, spline_evals = self.bspline_eval_nonzero(xvec)

        # Create 4^d hypercube of bspline products
        # Reduce uses left to right, transposes sound ugly, just reverse.
        eval_prods = reduce(lambda a, b: np.array([a*x for x in b]),
                            spline_evals[::-1])

        # This slice can be passed to a numpy grid of spline coefficients
        # to extract the 4^d relevant coefficients
        sl_base = tuple( slice(i0, i0+4, None) for i0 in imin_vals )

        # We will have arrays of shape (n_EI, n1, n2, ..., nd) where n_EI
        # is the number of empirical nodes, and n1, ..., nd are the number of
        # spline coefficients (2 + the number of grid points) in each dimension.
        # We want to sum over everything except the first index to evaluate the
        # n_EI different splines.
        summed_axes = tuple( i+1 for i in range(self.dim) )

        # Build a single slice so we can make better use of np.sum:
        # NOTE: sl object is meant to be used with spline coefficients
        # for multiple EIM nodes
        sl = tuple( itertools.chain([slice(None)], sl_base) )

        #self.last_return = [eval_prods, sl, summed_axes]

        return eval_prods, sl, summed_axes

#-----------------------------------------------------------------------------

# TODO: maybe ts_grid(x) should return sl_base and build sl here.
# the sl oject should also get cached
def fast_tensor_spline_eval(x,ts_grid,spline_coeffs):
    """ Evaluate SPLINE_COEFFS defined on the grid TS_GRID
        at an n-dimensional value X. """

    # See TensorSplineGrid's call method for documentation
    eval_prods, sl, summed_axes =  ts_grid(x)

    return np.sum(spline_coeffs[sl] * eval_prods, axis=summed_axes)

def fast_complex_tensor_spline_eval(x,ts_grid,spline_coeffs_real,spline_coeffs_imag):
    """ Evaluate SPLINE_COEFFS_REAL and SPLINE_COEFFS_IMAG defined on the 
        grid TS_GRID at an n-dimensional value X. """

    nre = fast_tensor_spline_eval(x,ts_grid,spline_coeffs_real)
    nim = fast_tensor_spline_eval(x,ts_grid,spline_coeffs_imag)

    return nre + 1.j*nim

