# Welcome to GWSurrogate! #

GWSurrogate is an easy to use interface to gravitational wave surrogate models.

Surrogates provide a fast and accurate evaluation mechanism for gravitational
waveforms which would otherwise be found through solving differential
equations. These equations must be solved in the ``building" phase, which
was performed using other codes. For details see

[1] Scott Field, Chad Galley, Jan Hesthaven, Jason Kaye, and Manuel Tiglio.
`"Fast prediction and evaluation of gravitational waveforms using surrogate
models". Phys. Rev. X 4, 031006 (2014). arXiv: gr-qc:1308.3565

If you find this package useful in your work, please cite reference [1] and,
if available, the relevant paper describing the specific surrogate used.

All available models can be found in gwsurrogate.catalog.list()

gwsurrogate is available at https://pypi.python.org

# Installation #

## Dependency ##

gwsurrogate requires:

1)  gwtools. If you are installing gwsurrogate with pip you
will automatically get gwtools. If you are installing gwsurrogate from
source, please see https://bitbucket.org/chadgalley/gwtools/

2) gsl. For speed, the long (hybrid) surrogates use gsl's spline function.
To build gwsurrogate you must have gsl installed. Fortunately, this is a
common library and can be easily installed with a package manager.

Note that at runtime (ie when you do import gwsurrogate) you may need to let
gsl know where your BLAS library is installed. This can be done by setting
your LD_PRELOAD or LD_LIBRARY_PATH environment variables. A relevant example:

```
>>> export LD_PRELOAD=~/anaconda3/envs/python27/lib/libgslcblas.so
```

## From pip ##

The python package pip supports installing from PyPI (the Python Package
Index). gwsurrogate can be installed to the standard location
(e.g. /usr/local/lib/pythonX.X/dist-packages) with

```
>>> pip install gwsurrogate
```

If there is no binary/wheel package already available for your operating system, the installer will
try to build the package from the sources. For that, you would need to have `gsl` installed already.
The installer will look for `GSL` inside `/opt/local/`. You may provide additional paths with the
`CPPFLAGS` and `LDFLAGS` environment variables. 

In the case of an `homebrew` installation, you may install the package like this:

```
>>> export HOMEBREW_HOME=`brew --prefix`
>>> 
>>> export CPPFLAGS="-I$HOMEBREW_HOME/include/"
>>> export LDFLAGS="-L$HOMEBREW_HOME/lib/"
>>> pip install gwsurrogate
```



## From conda ##

`gwsurrogate` is [on conda-forge](https://anaconda.org/conda-forge/gwsurrogate), and can be installed with

```
>>> conda install -c conda-forge gwsurrogate
```

## From source ##

First, please make sure you have the necessary dependencies
installed (see above). Next, Download and unpack gwsurrogate-X.X.tar.gz
to any folder gws_folder of your choosing. The gwsurrogate module can
be used immediately by adding

```
import sys
sys.path.append('absolute_path_to_gws_folder')
```

at the beginning of any script/notebook which uses gwsurrogate.

Alternatively, if you are a bash or sh user, edit your .profile
(or .bash_profile) file and add the line

```
export PYTHONPATH=~absolute_path_to_gws_folder:$PYTHONPATH
```

For a "proper" installation

```
>>> python setup.py install    # option 1
>>> pip install -e gwsurrogate # option 2
```

where the "-e" installs an editable (development) project with pip. This allows
your local code edits to be automatically seen by the system-wide installation.

If you have git cloned this project, you must do

```
git submodule init
git submodule update
```

# Usage #

## Available models
To get a list of all available surrogate models, do:
```python
>>> import gwsurrogate
>>> gwsurrogate.catalog.list()
>>> gwsurrogate.catalog.list(verbose=True)      # Use this for more details
```

### Current NR models
The most up-to-date models trained on numerical relativity data are listed below, along with links to example
notebooks.
- [NRSur7dq4](https://github.com/sxs-collaboration/gwsurrogate/blob/master/tutorial/website/NRSur7dq4.ipynb):
  For generically precessing BBHs, trained on mass ratios q≤4. Paper:
  [arxiv:1905.09300](https://arxiv.org/abs/1905.09300).
- [NRHybSur3dq8](https://github.com/sxs-collaboration/gwsurrogate/blob/master/tutorial/website/NRHybSur3dq8.ipynb):
  For nonprecessing BBHs, trained on mass ratios q≤8. Paper:
  [arxiv:1812.07865](https://arxiv.org/abs/1812.07865).
- [NRHybSur2dq15](https://github.com/sxs-collaboration/gwsurrogate/blob/master/tutorial/website/NRHybSur2dq15.ipynb):
  For nonprecessing BBHs, trained on q≤15, chi1≤0.5, chi2=0. Paper:
  [arxiv:2203.10109](https://arxiv.org/abs/2203.10109).
- [NRHybSur3dq8_CCE](https://github.com/sxs-collaboration/gwsurrogate/blob/master/tutorial/website/NRHybSur3dq8_CCE.ipynb):
  For nonprecessing BBHs, trained on CCE (Cauchy-characteristic evolution) waveforms of mass ratios q≤8. Unlike all of the other models, NRHybSur3dq8_CCE includes memory effects. Paper:
  [arxiv:2306.03148](https://arxiv.org/abs/2306.03148).
  
### Current point-particle blackhole perturbation theory models
The most up-to-date models trained on point-particle blackhole perturbation data and calibrated to numerical relativity (NR) in the comparable mass regime.
- [BHPTNRSur1dq1e4](https://github.com/sxs-collaboration/gwsurrogate/blob/master/tutorial/website/BHPTNRSur1dq1e4.ipynb):
  Nonspinning BBHs, trained on mass ratios q≤10000 and harmonics up to ell=10. Paper:
  [arxiv:2204.01972](https://arxiv.org/abs/2204.01972).

## Download surrogate data and load it
Pick a model, let's say `NRSur7dq4` and download the data. Note this only
needs to be done once.
```python
gwsurrogate.catalog.pull('NRSur7dq4')       # This can take a few minutes
```

Load the surrogate, this only needs to be done once at the start of a script
```python
sur = gwsurrogate.LoadSurrogate('NRSur7dq4')
```

## Evaluate the surrogate
```python
q = 4           # mass ratio, mA/mB >= 1.
chiA = [-0.2, 0.4, 0.1]         # Dimensionless spin of heavier BH
chiB = [-0.5, 0.2, -0.4]        # Dimensionless of lighter BH
dt = 0.1                        # timestep size, Units of total mass M
f_low = 0                # initial frequency, f_low=0 returns the full surrogate

# h is dictionary of spin-weighted spherical harmonic modes
# t is the corresponding time array in units of M
# dyn stands for dynamics, do dyn.keys() to see contents
t, h, dyn = sur(q, chiA, chiB, dt=dt, f_low=f_low)
```

There are many more options, such as using MKS units, returning the
polarizations instead of the modes, etc.  Read the documentation for more
details.
```python
help(sur)
```

Jupyter notebooks located in
[tutorial/website](https://github.com/sxs-collaboration/gwsurrogate/blob/master/tutorial/website)
give a more comprehensive overview of individual models.


# Tests #

If you have downloaded the entire project as a tar.gz file, its a good idea
to run some regression tests. 


```
>>> cd test                              # move into the folder test
>>> python download_regression_models.py # download all surrogate models to test
>>> python test_model_regression.py      # (optional - if developing a new test) generate regression data locally on your machine
>>> cd ..                                # move back to the top-level folder
>>> pytest                               # run all tests
>>> pytest -v -s                         # run all tests with high verbosity
```

# NSF Support #

This package is based upon work supported by the National Science Foundation
under PHY-1316424, PHY-1208861, and PHY-1806665.

Any opinions, findings, and conclusions or recommendations expressed in
gwsurrogate are those of the authors and do not necessarily reflect the
views of the National Science Foundation.
