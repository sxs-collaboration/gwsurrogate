---
title: 'GWSurrogate: A Python package for gravitational wave surrogate models'
tags:
  - Python
  - physics
  - general relativity
  - black holes
  - gravitational waves
authors:
  - name: Scott E. Field
    orcid: 0000-0002-6037-3277
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Vijay Varma
    orcid: 0000-0002-9994-1761
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Leo C. Stein
    orcid: 0000-0001-7559-9597
    affiliation: "2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Mathematics and Center for Scientific Computing \& Visualization Research, University of Massachusetts, Dartmouth, MA 02747
   index: 1
 - name: Department of Physics and Astronomy, The University of Mississippi, University, MS 38677, USA
   index: 2
date: 27 March 2024
bibliography: paper.bib

---

# TODOs (for paper writers)

* Do we need a more formal documentation like Read the Docs?

* Older models do not follow LVK conventions. Should this be updated before releasing the paper?

* Do we want some splashy figure?

* More citations (but lets not get too carried away as its not a review)

* Any other high-level stuff to add to the paper?

# Summary


Gravitational waves are ripples in space-time caused by the motion of massive objects. One of the most astrophysically important sources of gravitational radiation is caused by two orbiting compact objects, such as black holes and neutron stars, that slowly inspiral and merge. The motion of these massive objects generates gravitational waves that radiate to the far field where gravitational-wave detectors can observe them. Complicated partial or ordinary differential equations govern the entire process. Traditionally, the dynamics of compact binary systems and the emitted gravitational waves have been computed by expensive simulation codes that can take days to months to run. A key simulation output is the gravitational wave signal for a particular set of parameter values describing the system, such as the black holes' masses and spins.  The computed signal is required for a diverse range of multiple-query applications, such as template bank generation for searches, parameter estimation, mock data analysis, studies of model bias, and tests of general relativity, to name a few. In such settings, the high-fidelity signal computed from differential equations is often too slow to be directly used.

Surrogate models offer a practical way to dramatically accelerate model evaluation while retaining the high-fidelity accuracy of the expensive simulation code. Surrogate models can be constructed in various ways, but what separates these models from others is that they are primarily data-driven. Given a training set of gravitational waveform data sampling the parameter space, a model is built by following three steps: (i) Feature extraction: the waveform is decomposed into *data pieces* that are simple to model, (ii) Dimensionality reduction: each data piece is approximated by a low-dimensional vector space, which reduces the degrees of freedom we need to model, and (iii) regression techniques are applied to the low-dimensional representation of each data piece over the parameter space defining the model. These model-building steps result in an HDF5 file defining the surrogate model's data and structure, which is stored on Zenodo. The GWSurrogate code provides access to these HDF5 files through its catalog interface, and all available models and their HDF5 files can be found in `gwsurrogate.catalog.list()`. For a recent overview of surrogate modeling as used in gravitational wave astrophysics, please see Section 5 of @LISAConsortiumWaveformWorkingGroup:2023arg.

The development of ``GWSurrogate`` is hosted on [GitHub](https://github.com/sxs-collaboration/gwsurrogate) and distributed through both [PyPI](https://pypi.org/project/gwsurrogate/) and [Conda](https://anaconda.org/conda-forge/gwsurrogate/). Quick start guides are found on the project's [homepage](https://github.com/sxs-collaboration/gwsurrogate) while model-specific documentation is described through a collection of model-specific [Jupyter notebooks](https://github.com/sxs-collaboration/gwsurrogate/tree/master/tutorial). Automated testing is run on [GitHub Actions](https://github.com/sxs-collaboration/gwsurrogate/actions).


# Statement of need

``GWSurrogate`` is a Python package that provides an easy to use interface to gravitational wave surrogate models built 
using the methods described in (ADD CITATIONS HERE). More specifically, ``GWSurrogate`` gravitational wave models provide evaluation of
$$
\begin{align}
 h_{\tt S}(t, \theta, \phi;{\bf \Lambda}) = \sum^{\infty}_{\ell=2} \sum_{m=-\ell}^{\ell} h_{\tt S}^{\ell,m}(t;{\bf \Lambda}) ~^{-2}Y_{\ell m}(\theta, \phi) \,,
\end{align}
$$
where $^{-2}Y_{\ell m}$ are the spin$=-2$ weighted spherical harmonics and ${\bf \Lambda}$ describes the model's parameterization. The surrogate model provides fast evaluations for the modes, $h_{\tt S}^{\ell,m}$. As described more fully in the documentation, the high-level API allows users direct access to the modes $\{h_{\tt S}^{\ell,m}(t)\}$ (as a Python dictionary) or assembles the sum $h_{\tt S}(t, \theta, \phi)$ at a particular location $(\theta, \phi)$. The models implemented in ``GWSurrogate`` are intended to be used in production data analysis efforts. As such, computationally expensive operations (e.g., interpolation onto uniform time grids) are implemented by wrapping low-level C code for speed, whereas ``GWSurrogate`` provides a user-friendly interface to the higher-level waveform evaluation API. Models implemented in ``GWSurrogate`` also use the same waveform conventions of the LIGO-Virgo-Kagra collaboration, thus ensuring that downstream data analysis codes can use ``GWSurrogate`` models without needing to worry about pesky issues of matching conventions. ``GWSurrogate`` models can be directly evaluated in either physical units (as are often used in data analysis studies) and dimensionless units where all dimensioned quantities are expressed in terms of the system's total mass (as are often used in theoretical studies). 


# Acknowledgements

We acknowledge... 

# References