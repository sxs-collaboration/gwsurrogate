#!/usr/bin/env python

from setuptools import setup
from setuptools import Extension
import numpy

short_desc = "Utils required for precessing NR surrogates"
long_desc = \
"""
Utils required for speeding up precessing NR surrogates.
This includes the following:
    ODE integration required for the dynamics surrogates.
    Fit evaluations in the dynamics and coorbital surrogate.
"""

extensions = [
    Extension(
                '_utils',
                sources=['src/precessing_utils.c'],
                include_dirs = ['include', numpy.get_include()],
                language='c',
                extra_compile_args = ['-fPIC', '-O3'],
            )
        ]

setup(
        name            = 'precessing_utils',
        version         = '1.0.1',
        description     = short_desc,
        long_description = long_desc,
        author          = 'Vijay Varma, Jonathan Blackman',
        author_email    = 'vvarma@caltech.edu',
        ext_modules     = extensions,
    )
