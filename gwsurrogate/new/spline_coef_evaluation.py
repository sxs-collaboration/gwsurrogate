import numpy as np
import scipy.sparse.linalg
import os

class UniformSpacingCubicSplineND:
    """Computes coefficients for an N-dimensional spline
    on uniformly-spaced grid data.
    The spacing and grid size may be different in each dimension."""

    Bulk_coefs = [(-1, 1./6.), (0, 2./3.), (1, 1./6.)]
    Inner_edge_coefs = [(-1, 1./4.), (0, 7./12.), (1, 1./6.)]
    Outer_edge_coefs = [(-1, 1.)]
    Bc_coefs_thirdDeriv = [(0, 1.), (1, -2.), (2, 1.5), (3, -2./3.), (4, 1./6.)]
    Bc_coefs_natural = [(0, 1.), (1, -1.5), (2, 0.5)]


    def __init__(self, dimensions, origin=None, spacings=None):
        """
dimensions:
    (nx_1, nx_2, ...nx_N), number of points per dimension
origin:
    Specify the coordinates of the corner data point [x0_1, x0_2, ..., x0_N].
    None (default) becomes [0, ..., 0].
spacings:
    Specify the positive spacings [dx_1, dx_2, ..., dx_N].
    None (default) becomes [1., ..., 1.].
    """
        self.dims = dimensions
        self.nCoefs = np.array([n+2 for n in dimensions])
        self.N = len(dimensions)
        self.nTotal = np.prod(self.nCoefs)
#        self.coefs = np.zeros(dimensions)
        if origin is None:
            self.origin = np.zeros(self.N)
        else:
            self.origin = np.array(origin)
        if spacings is None:
            self.spacings = np.ones(self.N)
        else:
            self.spacings = np.array(spacings)
        self.lu = None # Full L,U decomposition of matrix


    def decompose(self, bdry_cond='third_deriv'):
        """
Finds the full LU decomposition of the matrix relating the spline
coefficients to the data on the grid.  This is the computationally
expensive step for large grids, and allows the spline coefficients
to be determined quickly given a new set of data.

bdry_cond: The boundary conditions to use for the tensor product spline.
    'third_deriv' (default): on a (N-1) dimensional boundary surface,
        sets the (constant) third partial derivative in the direction
        of the surface normal to be continuous across the boundary between
        the first and second intervals.
        This has the best results near the boundary for simple test cases.
    'natural': The natural spline termination condition sets the second
        partial derivative in the direction of the surface normal to be
        zero at the boundary grid points.
        Seen to have reasonably good performance near the boundary in
        simple test cases.
        """

        matrix_i, matrix_j, matrix_data = self._get_sparse_matrix(bdry_cond=bdry_cond)
        self._decompose(matrix_i, matrix_j, matrix_data)

    def _get_sparse_matrix(self, bdry_cond='third_deriv'):
        matrix_i = []
        matrix_j = []
        matrix_data = []
        if bdry_cond == 'third_deriv':
            bc_coefs = UniformSpacingCubicSplineND.Bc_coefs_thirdDeriv
        elif bdry_cond == 'natural':
            bc_coefs = UniformSpacingCubicSplineND.Bc_coefs_natural
        else:
            raise ValueError("Bad bdry_cond %s"%bdry_cond)

        # Store as list for ease of access
        forward_coefs = [bc_coefs, UniformSpacingCubicSplineND.Outer_edge_coefs, UniformSpacingCubicSplineND.Inner_edge_coefs, UniformSpacingCubicSplineND.Bulk_coefs]
        reversed_coefs = [None, [(-i, x) for i, x in bc_coefs],
                [(-i, x) for i, x in UniformSpacingCubicSplineND.Outer_edge_coefs], [(-i, x) for i, x in UniformSpacingCubicSplineND.Inner_edge_coefs]]

        # Coordinates of the next matrix_i
        i_indices = np.array([0 for _ in range(self.N)])

        # Coordinates of the next matrix_j are i_indices + [c[0] for c in coefs]
        # The matrix at the next matrix_i, matrix_j indices is prod([c[1] for c in coefs])
        coefs = [bc_coefs for _ in range(self.N)]

        while np.all(i_indices < self.nCoefs):
            # Index each of the coefs lists
            delta_indices = np.array([0 for _ in coefs])
            nc = np.array([len(c) for c in coefs])

            # Matrix first index associated with the coordinate
            mi_idx = 0
            for i in range(self.N):
                mi_idx = mi_idx*self.nCoefs[i] + i_indices[i]

            while np.all(delta_indices < nc):

                # Matrix second index associated with the coordinate
                mj_idx = 0
                for i, dii in enumerate(delta_indices):
                    mj_idx = mj_idx*self.nCoefs[i] + (i_indices[i] + coefs[i][dii][0])
                matrix_i.append(mi_idx)
                matrix_j.append(mj_idx)
                matrix_data.append(np.prod([coefs[i][dii][1] for i, dii in enumerate(delta_indices)]))

                # Increment last index of delta_indices
                j = self.N - 1
                delta_indices[j] += 1
                while delta_indices[j] >= nc[j] and j > 0:
                    delta_indices[j] = 0
                    j -= 1
                    delta_indices[j] += 1

            # Increment last index of i_indices, carrying over as necessary and adjusting coefs
            i = self.N - 1
            i_indices[i] += 1
            if i_indices[i] <= 3:
                coefs[i] = forward_coefs[i_indices[i]]
            elif i_indices[i] >= self.nCoefs[i] - 3:
                coefs[i] = reversed_coefs[self.nCoefs[i] - i_indices[i]]
            while i_indices[i] >= self.nCoefs[i] and i > 0:
                i_indices[i] = 0
                coefs[i] = bc_coefs
                i -= 1
                i_indices[i] += 1
                if i_indices[i] <= 3: 
                    coefs[i] = forward_coefs[i_indices[i]] 
                elif i_indices[i] >= self.nCoefs[i] - 3: 
                    coefs[i] = reversed_coefs[self.nCoefs[i] - i_indices[i]]

        return matrix_i, matrix_j, matrix_data

    def _decompose(self, matrix_i, matrix_j, matrix_data):
        self.lu = scipy.sparse.linalg.splu(scipy.sparse.csc_matrix((matrix_data, (matrix_i, matrix_j)), shape=(self.nTotal, self.nTotal)))

    def solve(self, griddata):
        if not np.shape(griddata) == self.dims:
            raise ValueError("griddata should have shape {}".format(self.dims))

        if self.lu is None:
            print "First doing decomposition..."
            self.decompose()

        padded_data = append_end_zeros(griddata)
        self.coefs = self.lu.solve(padded_data.flatten()).reshape(np.shape(padded_data))


    def saveDecomposition(self, dirname):
        """Saves all the decomposition data in a new directory."""

        if self.lu is None:
            raise Exception("No point in saving an undecomposed spline")
        if os.path.exists(dirname):
            raise Exception("dirname already exists!")
        os.mkdir(dirname)
        for idstr, sm in zip(["L", "U"], [self.lu.L, self.lu.U]):
            np.save(dirname + "/" + idstr + "_data.npy", sm.data)
            np.save(dirname + "/" + idstr + "_indices.npy", sm.indices)
            np.save(dirname + "/" + idstr + "_indptr.npy", sm.indptr)
        np.save(dirname + "/perm_c.npy", self.lu.perm_c)
        np.save(dirname + "/perm_r.npy", self.lu.perm_r)

def append_end_zeros(data):
    if len(np.shape(data)) > 1:
        tmp = np.array([append_end_zeros(d) for d in data])
        zero = np.array([0.*tmp[0]])
        return np.append(zero, np.append(tmp, zero, 0), 0)
    return np.append(0., np.append(data, 0.))
