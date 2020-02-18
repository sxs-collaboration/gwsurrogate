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
import numpy
extmod =  Extension('gwsurrogate.precessing_utils._utils',
                    sources=['gwsurrogate/precessing_utils/src/precessing_utils.c'],
                    include_dirs = ['gwsurrogate/precessing_utils/include', numpy.get_include()],
                    language='c',
                    extra_compile_args = ['-std=c99','-fPIC', '-O3'])
extmods.append(extmod)

# Extract code version from surrogate.py
def read_main_file(key):
    with open('gwsurrogate/surrogate.py') as f:
        for line in f.readlines():
            if key in line:
                return line.split('"')[1]

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
      install_requires=["gwtools"],
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
      ext_modules = extmods,
)
