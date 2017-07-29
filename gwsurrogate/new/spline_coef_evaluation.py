"""
Module for evaluating cubic tensor spline coefficients given the data
to be interpolated on a uniformly spaced tensor product grid.

Apologies for not writing this up in LaTeX. Hopefully nobody will ever need
to read it and the code will just work.

Suppose we have a d-dimensional grid with dimensions (N1 x N2 x ... x Nd).
Denote x_{i1, i2, ..., id} as the coordinates of the grid point with indices
[i1, i2, ..., id]. We are interested in interpolating some function f(x),
where x is a d-dimensional vector and f is real valued. Denote the grid data
by f_{i1, ..., id} = f(x_{i1, ..., id}), which is known.

A cubic tensor spline
interpolant for f requires a grid of ((N1 + 2) x (N2 + 2) x ... x (Nd + 2))
spline coefficients. We can extend the coordinate grid to a "coefficient grid"
by adding one additional grid point on both boundaries of each grid dimension.
We pad the grid data on these boundary points with zeros, and we will need to
impose boundary conditions to determine these coefficients.

The spline interpolant evaluated at x = (x1, ..., xd) is of the form
  TS(x) = \sum_{j1, ..., jd} c_{j1, ..., jd} s1_{j1}(x1) ... sd_{jd}(xd)
where the index ji runs over the (Ni + 2) spline coefficient indices,
c_{j1, ..., jd} are the spline coefficients, and si_{ji} is a one-dimensional
cubic BSpline associated with point ji in the j'th dimension of the coefficient
grid. Evaluating this on the spline coefficient grid gives us
((N1+2) x ... x (Nd+2)) equations:
    f_{i1, ..., id} =
        \sum_{j1, ..., jd} c_{j1, ..., jd} M1_{i1, j1} ... Md_{id, jd}
where Mk_{ik, jk} is a ((Nk+2) x (Nk+2)) matrix giving the BSpline associated
with point jk of the k'th dimension of the coefficient grid evaluated at the
point ik of the k'th dimension of the *padded* coordinate grid. When ik is one
of the boundary points (0 or Nk+1), we are free to set Mk_{ik, jk} as we wish
to control the boundary conditions which will result when we solve this system
of equations. 

The matrices Mk are exactly those used when solving for coefficients of a
one-dimensional cubic spline! For example, with 'not-a-knot' (constant third
derivative) boundary conditions, we have:

M= [[1,    -2,    3/2,   -2/3,    1/6,      0,      0,      0,      0],
    [1,     0,      0,      0,      0,      0,      0,      0,      0],
    [0,   1/4,   7/12,    1/6,      0,      0,      0,      0,      0],
    [0,     0,    1/6,    2/3,    1/6,      0,      0,      0,      0],
    [0,     0,      0,    1/6,    2/3,    1/6,      0,      0,      0],
    [0,     0,      0,      0,    1/6,    2/3,    1/6,      0,      0],
    [0,     0,      0,      0,      0,    1/6,   7/12,    1/4,      0],
    [0,     0,      0,      0,      0,      0,      0,      0,      1],
    [0,     0,      0,      0,    1/6,   -2/3,    3/2,     -2,      1]]

Here, the first and last rows represent the 'not-a-knot' boundary condition.
If we then look at column j, ignoring the first and last rows we have the
j'th BSpline evaluated at each of the Nk grid points.
The first three and last three columns differ from the middle (bulk) columns
because the spline breakpoints are not uniformly spaced near the boundary and
so the BSplines near the boundary are different.
For example, this (9x9) matrix corresponds to a grid of 7 uniformly spaced
points which we will take to be [0, 1, 2, 3, 4, 5, 6].
The spline breakpoints would then be [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6].
The first spline coefficient corresponds to a cubic BSpline with breakpoints
[0, 0, 0, 0, 1], which is 0 outside of [0, 1), and rolls smoothly from 1 at x=0
to 0 at x=1. So this BSpline is 1 at the first grid point and 0 at all other
grid points, giving us the first column of the matrix (ignoring the first and
last rows). The second spline coefficient corresponds to a cubic BSpline with
breakpoints [0, 0, 0, 1, 2]. This BSpline 1/4 at x=1 and 0 at all other grid
points, giving us the second column (again neglecting the first and final rows).
The third column also corresponds to a boundary BSpline, but once we reach the
fourth column we have a BSpline with breakpoints [0, 1, 2, 3, 4] which is a
"typical" or "bulk" case - the breakpoints are uniformly spaced, and all "bulk"
BSplines are identical up to a translation. This BSpline has support [0, 4),
is 0 at the boundaries of its support, and takes on the values 1/6, 2/3, and 1/6
at x=1, x=2, and x=3 respectively.

Okay, now we are ready to solve our HUGE ((N1+2) x ... x (Nd+2)) linear system
of equations without too much pain. We first find inverses of the Mk matrices,
denoted MkInv, such that \sum_{i} MkInv{n, i} Mk_{i,j} = \delta_{n, j}.
We can then compute
    \sum_{i1, ..., id} f_{i1, ..., id} M1Inv{n1, i1} ... MdInv{nd, id} =
    \sum_{j1, ..., jd} c_{j1, ..., jd} \delta_{n1, j1} ... \delta{nd, jd} =
    c_{n1, ..., nd}
So we can solve for all the coefficients just by summing our padded grid data
against all of these (small) matrices! Of course, since we have many indices,
this requires some nice numpy tensordot magic :)
"""

import numpy as np

COEFS = {
    "bulk": [(-1, 1./6.), (0, 2./3.), (1, 1./6.)],
    "inner edge": [(-1, 1./4.), (0, 7./12.), (1, 1./6.)],
    "outer edge": [(-1, 1.)],
    "BC not-a-knot": [(0, 1.), (1, -2.), (2, 1.5), (3, -2./3.), (4, 1./6.)],
    "BC natural": [(0, 1.), (1, -1.5), (2, 0.5)],
        }

def get_1d_spline_matrix(n, bc="not-a-knot"):
    """
    Constructs the matrix A such that A*c = f, where c is a vector of spline
    coefficients and f is the solution evaluated on the knots, padded with one
    zero on either side for boundary conditions.

    n: The number of spline coefficients (number of knots + 2)
    bc: The boundary condition. Can be "not-a-knot" (constant 3rd derivative)
        or "natural".
    """
    if n < 6:
        raise Exception("Not implemented for < 4 knots, using cubic splines!")
    matrix = np.zeros((n, n))
    bc_coef_key = "BC %s"%(bc)
    if not bc_coef_key in COEFS.keys():
        raise Exception("Unknown boundary condition: %s"%(bc))

    # Start with the boundaries and work our way in
    for j, c in COEFS[bc_coef_key]:
        matrix[0, j] = c
        matrix[-1, -1-j] = c

    for j, c in COEFS["outer edge"]:
        matrix[1, 1+j] = c
        matrix[-2, -2-j] = c

    for j, c in COEFS["inner edge"]:
        matrix[2, 2+j] = c
        matrix[-3, -3-j] = c

    for i in range(3, n-3):
        for j, c in COEFS["bulk"]:
            matrix[i, i+j] = c
    return matrix


class UniformSpacingCubicSplineND:
    """
Computes coefficients for an N-dimensional cubic spline
on uniformly-spaced grid data.
    """

    def __init__(self, dimensions, BC='not-a-knot'):
        """
dimensions:
    (nx_1, nx_2, ...nx_d), number of points per dimension
BC:
    Boundary conditions for the splines. Can be 'not-a-knot' or 'natural'.
        """
        self.dims = dimensions
        self.d = len(dimensions)
        self.nCoefs = np.array([n+2 for n in dimensions])
        self.N = len(dimensions)
        self.nTotal = np.prod(self.nCoefs)
        self.BC = BC

        t_seconds = self.nTotal * 4.e-7 # ~ order of magnitude
        if t_seconds > .01:
            print('%s coefficients: solves should take O(%0.1e seconds) each.'%(
                self.nTotal, t_seconds))

        self.setup_1d_matrices()

    def setup_1d_matrices(self):
        """
Due to the tensor-product structure, we do not need to invert
the ((nx_1 * ... * nx_d) X (nx_1 * ... * nx_d)) system of
equations, and can instead solve a small system for each dimension.
Here we find the necessary inverse matrices.
        """
        self.inv_1d_matrices = []
        for n in self.nCoefs:
            matrix = get_1d_spline_matrix(n)
            # TODO: Better method for inverting?
            # These matrices are almost tridiagonal
            self.inv_1d_matrices.append(np.linalg.inv(matrix))

    def solve(self, griddata):
        """
Given the function evaluated on the knots, computes the tensor-spline
coefficients that can be used to interpolate the function.
        """
        if not np.shape(griddata) == self.dims:
            raise ValueError("griddata should have shape {}".format(self.dims))

        tmp_result = np.pad(griddata, 1, 'constant')

        # We will apply the 1d inverse matrices from last to first.
        # With each application, the shape of tmp_result goes from
        # (for example) (a, b, c, d) -> (d, a, b, c) -> (c, d, a, b)
        # so that we are always applying the 1d grid matrix to the last index.
        # We also end up with the correct shape at the end and don't have to
        # worry about multidimensional transposes.
        for minv in self.inv_1d_matrices[::-1]:
            tmp_result = np.tensordot(minv, tmp_result, (1, self.d - 1))

        return tmp_result
