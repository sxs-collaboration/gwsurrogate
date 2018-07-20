import sys

try:
  from setuptools import setup
  setup
except ImportError: # currently not supported
  raise ImportError("GWSurrogate requires setuptools to correctly install the gwools submodule")
  #from distutils.core import setup # currently not supported
  #setup

# To render markdown. See https://github.com/pypa/pypi-legacy/issues/148
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except ImportError:
    long_description = open('README.md').read()


setup(name='gwsurrogate',
      version='0.8.3',
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
      # install_requires=["numpy","matplotlib","scipy"],
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
)
