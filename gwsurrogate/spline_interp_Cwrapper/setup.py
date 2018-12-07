# This extension should be built from the gwsurrogate-level setup.py
# script. However, this setup.py is here for local builds
from distutils.core import setup, Extension
import os

# If /opt/local directories exist, use them
if os.path.isdir('/opt/local/include'):
    IncDirs = ['/opt/local/include']
else:
    IncDirs = []

if os.path.isdir('/opt/local/lib'):
    LibDirs = ['/opt/local/lib']
else:
    LibDirs = []

extmod = Extension('_spline_interp',
                   include_dirs = IncDirs,
                   libraries = ['gsl'],
                   library_dirs = LibDirs,
                   sources = ['_spline_interp.c'])

setup (ext_modules = [extmod])
