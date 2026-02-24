# This extension should be built from the gwsurrogate-level setup.py
# script. However, this setup.py is here for local builds
from distutils.core import setup, Extension

extmod = Extension('_spline_interp',
                   extra_compile_args=['-std=c++14'],
                   language="c++",
                   sources = ['_spline_interp.cpp'])

setup (ext_modules = [extmod])
