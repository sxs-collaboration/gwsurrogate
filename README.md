[![PyPI version](https://badge.fury.io/py/gwsurrogate.svg)](https://pypi.org/project/gwsurrogate/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gwsurrogate.svg)](https://anaconda.org/conda-forge/gwsurrogate)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07073/status.svg)](https://doi.org/10.21105/joss.07073)
[![arXiv:2504.08839](https://img.shields.io/badge/arXiv-2504.08839-B31B1B.svg)](https://arxiv.org/abs/2504.08839)


# Welcome to GWSurrogate! #

GWSurrogate is an easy to use interface to gravitational wave surrogate models.

Surrogates provide a fast and accurate evaluation mechanism for gravitational
waveforms which would otherwise be found through solving differential
equations. These equations must be solved in the ``building" phase, which
was performed using other codes. 

If this package contributes to a project that leads to a publication, 
please acknowledge this by citing the relevant paper(s). Please see the 
[How to Cite](https://github.com/sxs-collaboration/gwsurrogate/edit/master/README.md#how-to-cite) section at the bottom of this README file. 

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

## numpy 1.x and 2.x ##

Certain gwsurrogate modules are implemented as C-extensions and require NumPy’s C-API headers at build time. By default, `pip install .` uses the NumPy 2.x headers (as pinned in **pyproject.toml**) but produces binaries that remain compatible with NumPy 1.x at runtime. If you explicitly need to build against NumPy 1.x headers, update the NumPy requirement in **pyproject.toml** before installing.

To create a Conda environment with Python 3.11 and NumPy < 2.0:

```bash
conda create -n myenv python=3.11 "numpy<2.0"
```

## From source (pip) ##

First, please ensure you have the necessary dependencies
installed (see above). Next, git clone this project, to any
folder of your choosing. Then run

```
git submodule init
git submodule update
```

For a "proper" installation, run the following commands from the top-level gwsurrogate folder containing setup.py

```
>>> python -m pip install .            # option 1
>>> python -m pip install --editable . # option 2
```

where the "--editable" installs an editable (development) project with pip. This allows
your local code edits to be automatically seen by the system-wide installation.


# Documentation
 
 Explore our [Jupyter Notebooks](https://github.com/sxs-collaboration/gwsurrogate/blob/master/tutorial/website) for a comprehensive overview of individual models and the user-level API. For an introductory explanation of the surrogate modeling methodology used in GWSurrogate, check out these videos:

  - [Introduction to the GWSurrogate package](https://icerm.brown.edu/video_archive/2413)
  - [Overview of surrogate modeling methodology](https://icerm.brown.edu/video_archive/2412)


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
The most up-to-date models are trained on point-particle blackhole perturbation data and calibrated to numerical relativity (NR) in the comparable mass regime.
- [BHPTNRSur1dq1e4](https://github.com/sxs-collaboration/gwsurrogate/blob/master/tutorial/website/BHPTNRSur1dq1e4.ipynb):
  Nonspinning BBHs, trained on mass ratios q≤10000 and harmonics up to ell=10. Paper:
  [arxiv:2204.01972](https://arxiv.org/abs/2204.01972).

### Current effective one body models
The most up-to-date effective one body surrogate models.
- [SEOBNRv4PHMSur](https://github.com/sxs-collaboration/gwsurrogate/blob/master/tutorial/website/SEOBNRv4PHM.ipynb):
  precessing binary black hole with 2<=ell<=5 modes in inertial frame. Trained on mass ratios q ≤20. Paper:
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
q = 4                           # mass ratio, mA/mB >= 1.
chiA = [-0.2, 0.4, 0.1]         # Dimensionless spin of heavier BH
chiB = [-0.5, 0.2, -0.4]        # Dimensionless of lighter BH
dt = 0.1                        # timestep size, Units of total mass M
f_low = 0                       # initial frequency, f_low=0 returns the full surrogate

# optional parameters for a precessing surrogate models
precessing_opts = {'return_dynamics': True}

# h is dictionary of spin-weighted spherical harmonic modes
# t is the corresponding time array in units of M
# dyn stands for dynamics, do dyn.keys() to see contents
t, h, dyn = sur(q, chiA, chiB, dt=dt, f_low=f_low, precessing_opts=precessing_opts)
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

If you have git cloned this project and installed (and intalled it
using the `--editable` option), its a good idea to run some regression tests. 


```
>>> cd test                              # move into the folder test
>>> python download_regression_models.py # download all surrogate models to test
>>> python test_model_regression.py      # (optional - if developing a new test) generate regression data locally on your machine
>>> cd ..                                # move back to the top-level folder
>>> pytest                               # run all tests
>>> pytest -v -s                         # run all tests with high verbosity
```

# Contributing

We welcome contributions! Here's how you can get involved:

1. **Report Bugs or Suggest Enhancements**:  
   Use the [GitHub issue tracker](https://github.com/sxs-collaboration/gwsurrogate/issues) to report bugs or suggest new features. Before submitting, consider browsing through existing issues to see if your concern has already been addressed. A developer will respond to issues that are opened on GitHub.

2. **Contribute Code**:  
   We use the [fork and pull request model](https://help.github.com/articles/creating-a-pull-request-from-a-fork/) for code contributions. Fork the repository, make your changes, and submit a pull request. We use [Ruff](https://github.com/charliermarsh/ruff) for linting and auto-fixes. If you’re on VS Code, install the Ruff extension.

Please ensure you follow our [Code of Conduct](https://github.com/sxs-collaboration/gwsurrogate?tab=coc-ov-file) when contributing. 

# How to cite

If this package contributes to a project that leads to a publication, please acknowledge this by citing the GWSurrogate article in JOSS. The paper has the following bibtex entry

```
@article{Field:2025isp,
    author = "Field, Scott E. and Varma, Vijay and Blackman, Jonathan and Gadre, Bhooshan and Galley, Chad R. and Islam, Tousif and Mitman, Keefe and Pürrer, Michael and Ravichandran, Adhrit and Scheel, Mark A. and Stein, Leo C. and Yoo, Jooheon",
    title = "{GWSurrogate: A Python package for gravitational wave surrogate models}",
    eprint = "2504.08839",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.21105/joss.07073",
    journal = "J. Open Source Softw.",
    volume = "10",
    number = "107",
    pages = "7073",
    year = "2025"
}
````

Please also cite the relevant paper(s) describing your specific model. This information can be found by doing 

```python
>>> import gwsurrogate
>>> gwsurrogate.catalog.list(verbose=True) 
```


# NSF Support #

This package is based upon work supported by the National Science Foundation
under PHY-1316424, PHY-1208861, and PHY-1806665.

Any opinions, findings, and conclusions or recommendations expressed in
gwsurrogate are those of the authors and do not necessarily reflect the
views of the National Science Foundation.
