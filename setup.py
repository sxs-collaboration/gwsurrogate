import sys, os

try:
  from setuptools import setup, Extension
  setup
except ImportError: # currently not supported
  raise ImportError("GWSurrogate requires setuptools")
  #from distutils.core import setup # currently not supported
  #setup

# To render markdown. See https://github.com/pypa/pypi-legacy/issues/148
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except ImportError:
    long_description = open('README.md').read()

# all extensions here
extmods = []

# build extension 1: python wrapper to gsl's spline function
if os.path.isdir('/opt/local/include'):
    IncDirs = ['/opt/local/include']
else:
    IncDirs = []
if os.path.isdir('/opt/local/lib'):
    LibDirs = ['/opt/local/lib']
else:
    LibDirs = []
extmod = Extension('gwsurrogate.spline_interp_Cwrapper._spline_interp',
                   include_dirs = IncDirs,
                   libraries = ['gsl'],
                   extra_compile_args = ['-std=c99'],
                   library_dirs = LibDirs,
                   sources = ['gwsurrogate/spline_interp_Cwrapper/_spline_interp.c'])
extmods.append(extmod)

# build extension 2: precessing utils
extmod =  Extension('gwsurrogate.precessing_utils._utils',
                    sources=['gwsurrogate/precessing_utils/src/precessing_utils.c'],
                    include_dirs = ['gwsurrogate/precessing_utils/include'],
                    language='c',
                    extra_compile_args = ['-std=c99','-fPIC', '-O3'])
extmods.append(extmod)

# Workaround: Only import numpy once reqs have been imported
# Thanks to https://stackoverflow.com/a/42163080/1695428
from distutils.command.build_ext import build_ext
class LateNumpyIncludeCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)

# Extract code version from surrogate.py
def read_main_file(key):
    with open('gwsurrogate/surrogate.py') as f:
        for line in f.readlines():
            if key in line:
                return line.split('"')[1]

# define models to be used within pycbc
entries = {"pycbc.waveform.td":
            ["GWS-NRHybSur3dq8 = gwsurrogate.pycbc:gws_td_gen",
             "GWS-NRSur7dq4 = gwsurrogate.pycbc:gws_td_gen",
             "GWS-NRHybSur3dq8Tidal = gwsurrogate.pycbc:gws_td_gen",
            ]
          }

setup(name='gwsurrogate',
      version=read_main_file("__version__"),
      author=read_main_file("__author__"),
      author_email='sfield@umassd.edu',
      packages=['gwsurrogate'],
      license='MIT',
      include_package_data=True,
      contributors=[
      # Alphabetical by last name.
      ""],
      description='An easy to use interface to gravitational wave surrogate models',
      long_description_content_type='text/markdown',
      long_description=long_description,
      # will start new downloads if these are installed in a non-standard location
      install_requires=[
                "numpy",
                "scipy",
                "h5py",
                "scikit-learn",
                "gwtools",
                ],
      classifiers=[
                'Intended Audience :: Other Audience',
                'Intended Audience :: Science/Research',
                'Natural Language :: English',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Topic :: Scientific/Engineering :: Physics',
      ],
      entry_points = entries,
      cmdclass = {'build_ext': LateNumpyIncludeCommand},
      ext_modules = extmods,
)
