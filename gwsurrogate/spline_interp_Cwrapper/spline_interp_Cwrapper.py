import ctypes
from ctypes import c_double, c_long, POINTER, c_int
import numpy as np

from . import _spline_interp as _mod
_dll_path = _mod.__file__

def _load_c_func(dll_path, function_name, argtypes, restype=None):
    dll = ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
    func = getattr(dll, function_name)
    func.argtypes = argtypes
    if restype is not None:
        func.restype = restype
    return func

c_interp = _load_c_func(_dll_path, 'spline_interp', [
    c_long, c_long,
    POINTER(c_double), POINTER(c_double),
    POINTER(c_double), POINTER(c_double),
], restype=c_int)
c_interp_multi = _load_c_func(_dll_path, 'spline_interp_multi', [
    c_long, c_long, c_long,
    POINTER(c_double),
    POINTER(POINTER(c_double)),
    POINTER(c_double),
    POINTER(POINTER(c_double)),
], restype=c_int)
c_interp_multi_complex = _load_c_func(_dll_path, 'spline_interp_multi_complex', [
    c_long, c_long, c_long,
    POINTER(c_double),          # data_x
    POINTER(POINTER(c_double)), # data_y (interleaved re,im)
    POINTER(c_double),          # out_x
    POINTER(POINTER(c_double)), # out_y  (interleaved re,im)
], restype=c_int)

_SPLINE_ERRORS = {
    1: 'Memory allocation failed in spline interpolation',
    2: 'Evaluation point outside data range (extrapolation not supported)',
    3: 'Zero-length interval in input x data (duplicate or non-monotonic knots)',
    4: 'data_size must be >= 3 (need at least 3 knots for cubic spline)',
    5: 'out_size must be > 0 (no evaluation points provided)',
    6: 'num_datasets must be > 0 (no datasets provided)',
}

def _check_spline_rc(rc):
    if rc != 0:
        msg = _SPLINE_ERRORS.get(rc, 'Unknown spline error (code %d)' % rc)
        raise RuntimeError(msg)

def interpolate(xnew, x, y):
    x = x.astype('float64', copy=False)
    y = y.astype('float64')
    xnew = xnew.astype('float64', copy=False)

    x_p = x.ctypes.data_as(POINTER(c_double))
    y_p = y.ctypes.data_as(POINTER(c_double))
    xnew_p = xnew.ctypes.data_as(POINTER(c_double))

    ynew  = np.empty(xnew.shape[0])
    ynew_p = ynew.ctypes.data_as(POINTER(c_double))

    rc = c_interp(x.shape[0],xnew.shape[0],x_p,y_p,xnew_p,ynew_p)
    _check_spline_rc(rc)

    return ynew

def interpolate_many(xnew, x, y):
    """Interpolate multiple real-valued datasets sharing the same x-grid.

    Uses natural cubic spline interpolation (y'' = 0 at boundaries).

    Parameters
    ----------
    xnew : (M,) array_like, float64
        Evaluation points. Must lie within ``[x[0], x[-1]]`` (no
        extrapolation). Sorted ascending is optimal but not required.
    x : (N,) array_like, float64
        Knot x-coordinates, strictly monotonically increasing (no
        duplicates). Must have N >= 3.
    y : (num_datasets, N) array_like, float64
        Data values. Each row is an independent dataset sampled at `x`.
        Must be C-contiguous; will be copied if not already float64.

    Returns
    -------
    ynew : (num_datasets, M) ndarray, dtype float64
        Interpolated values at `xnew` for each dataset.

    Raises
    ------
    RuntimeError
        If the C interpolation routine returns a non-zero error code:
        out-of-bounds evaluation point, duplicate/non-monotonic knots,
        fewer than 3 knots, or memory allocation failure.
    """
    x = x.astype('float64', copy=False)
    y = np.ascontiguousarray(y, dtype='float64')
    xnew = xnew.astype('float64', copy=False)

    num_datasets, n_x = y.shape

    x_p = x.ctypes.data_as(POINTER(c_double))
    xnew_p = xnew.ctypes.data_as(POINTER(c_double))

    ynew = np.empty((y.shape[0], xnew.shape[0]), dtype='float64')

    n_datasets = y.shape[0]
    row_bytes  = y.shape[1]    * y.itemsize
    out_bytes  = xnew.shape[0] * ynew.itemsize

    PtrArr  = POINTER(c_double) * n_datasets
    y_ptrs    = PtrArr(*(ctypes.cast(y.ctypes.data    + d * row_bytes,
                                     POINTER(c_double)) for d in range(n_datasets)))
    ynew_ptrs = PtrArr(*(ctypes.cast(ynew.ctypes.data + d * out_bytes,
                                     POINTER(c_double)) for d in range(n_datasets)))

    rc = c_interp_multi(x.shape[0], xnew.shape[0], n_datasets, x_p, y_ptrs, xnew_p,
                        ynew_ptrs)
    _check_spline_rc(rc)

    return ynew

def interpolate_many_complex(xnew, x, y):
    """Interpolate multiple complex128 datasets sharing the same x-grid.

    Parameters
    ----------
    xnew : (M,) float64 array — evaluation points
    x    : (N,) float64 array — knot x-coordinates
    y    : (num_datasets, N) complex128 array — data values

    Returns
    -------
    ynew : (num_datasets, M) complex128 array
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.complex128)
    xnew = np.ascontiguousarray(xnew, dtype=np.float64)

    n_datasets, n_x = y.shape
    n_out = xnew.shape[0]

    x_p    = x.ctypes.data_as(POINTER(c_double))
    xnew_p = xnew.ctypes.data_as(POINTER(c_double))

    ynew = np.empty((n_datasets, n_out), dtype=np.complex128)

    # Each complex128 row is 2*n doubles interleaved (re,im,re,im,...)
    row_bytes = n_x  * 16  # sizeof(complex128) = 16
    out_bytes = n_out * 16

    PtrArr = POINTER(c_double) * n_datasets
    y_ptrs    = PtrArr(*(ctypes.cast(y.ctypes.data    + d * row_bytes,
                                     POINTER(c_double)) for d in range(n_datasets)))
    ynew_ptrs = PtrArr(*(ctypes.cast(ynew.ctypes.data + d * out_bytes,
                                     POINTER(c_double)) for d in range(n_datasets)))

    rc = c_interp_multi_complex(n_x, n_out, n_datasets, x_p, y_ptrs, xnew_p,
                                ynew_ptrs)
    _check_spline_rc(rc)

    return ynew
