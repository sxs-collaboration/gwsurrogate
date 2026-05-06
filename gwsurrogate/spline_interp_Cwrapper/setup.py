"""Local build script for the spline interpolation extension.

This extension should normally be built from the top-level ``setup.py`` script.
This local setup script is intended only for testing and diagnostics.

The build arguments here are not guaranteed to match the top-level setup.py, so
the resulting extension can behave differently. Users who run this local setup
script are responsible for setting the build arguments correctly for their
environment.
"""

from distutils.core import setup, Extension
import warnings


warnings.warn(
    "This extension should normally be built from the top-level setup.py. "
    "Only use gwsurrogate/spline_interp_Cwrapper/setup.py for local testing "
    "and diagnostics.",
    UserWarning,
    stacklevel=2,
)

extmod = Extension('_spline_interp',
                   extra_compile_args=['-std=c++14'],
                   language="c++",
                   sources = ['_spline_interp.cpp'])

setup (ext_modules = [extmod])
