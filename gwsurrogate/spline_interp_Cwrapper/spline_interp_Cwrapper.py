import ctypes
from ctypes import c_double, c_long, POINTER, util
import numpy as np
import os
from glob import glob

def _load_spline_interp(dll_path,function_name):

    cblas_path = util.find_library('cblas')
    if cblas_path is None:
        cblas_path = util.find_library('gslcblas')
        if cblas_path is None:
            raise OSError("Couldn't load libcblas or libgslcblas!")

    dllCBLAS = ctypes.CDLL(cblas_path, mode=ctypes.RTLD_GLOBAL)

    dll = ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
    func = dll.spline_interp
    func.argtypes = [c_long, c_long,
        POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double)]
    return func

dll_dir = os.path.dirname(os.path.realpath(__file__))
dll_path_glob = '%s/_spline_interp*so'%dll_dir
spline_libs = glob(dll_path_glob)
if len(spline_libs) == 0:
  all_files = glob('%s/*'%dll_dir)
  msg = '_spline_interp library not found! Searched in path %s which has files...\n'%dll_dir
  for all_file in all_files:
    msg += all_file+"\n"
  raise Exception(msg)
elif len(spline_libs) > 1:
  raise Exception('there should be only one _spline_interp library!')
else:
  #c_interp = _load_spline_interp('%s/_spline_interp.so'%dll_dir, 'spline_interp')
  c_interp = _load_spline_interp(spline_libs[0], 'spline_interp')

def interpolate(xnew, x, y):

    if min(xnew) < min(x) or max(xnew) > max(x):
        raise Exception('Extrapolation not allowed')

    x = x.astype('float64')
    y = y.astype('float64')
    xnew = xnew.astype('float64')

    x_p = x.ctypes.data_as(POINTER(c_double))
    y_p = y.ctypes.data_as(POINTER(c_double))
    xnew_p = xnew.ctypes.data_as(POINTER(c_double))

    ynew  = np.zeros(xnew.shape[0])
    ynew_p = ynew.ctypes.data_as(POINTER(c_double))

    c_interp(x.shape[0],xnew.shape[0],x_p,y_p,xnew_p,ynew_p)

    return ynew
