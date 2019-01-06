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

# build extension: python wrapper to gsl's spline function
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
                   library_dirs = LibDirs,
                   sources = ['gwsurrogate/spline_interp_Cwrapper/_spline_interp.c'])


setup(name='gwsurrogate',
      version='0.9.1',
      author='Jonathan Blackman, Scott Field, Chad Galley, Vijay Varma',
      author_email='sfield@umassd.edu',
      packages=['gwsurrogate'],
      license='MIT',
      include_package_data=True,
      contributors=[
      # Alphabetical by last name.
      ""],
      description='An easy to use interface to gravitational wave surrogate models',
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
      ext_modules = [extmod],
)
