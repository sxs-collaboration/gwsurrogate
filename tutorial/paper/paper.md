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
    affiliation: "1, 6"
  - name: Vijay Varma
    orcid: 0000-0002-9994-1761
    affiliation: "1"
  - name: Jonathan Blackman
    affiliation: "4"
  - name: Bhooshan Gadre
    orcid: 0000-0002-1534-9761
    affiliation: "3"
  - name: Chad R. Galley
    affiliation: "5"
  - name: Tousif Islam
    orcid: 0000-0002-3434-0084
    affiliation: "1, 3"
  - name: Gaurav Khanna
    orcid: 0000-0002-2565-8170
    affiliation: "6, 1"
  - name: Keefe Mitman
    orcid: 0000-0003-0276-3856
    affiliation: "7"
  - name: Michael Puerrer
    orcid: 0000-0002-3329-9788
    affiliation: "6"
  - name: Mark A. Scheel
    orcid: 0000-0001-6656-9134
    affiliation: "7"
  - name: Leo C. Stein
    orcid: 0000-0001-7559-9597
    affiliation: "2"
  - name: Jooheon Yoo
    orcid: 0000-0002-3251-0924
    affiliation: "3"
affiliations:
  - name: Department of Mathematics and Center for Scientific Computing \& Data Science Research, University of Massachusetts, Dartmouth, MA 02747, USA
    index: 1
  - name: Department of Physics and Astronomy, The University of Mississippi, University, MS 38677, USA
    index: 4
  - name: Kavli Institute for Theoretical Physics, University of California Santa Barbara, CA 93106, USA
    index: 3
  - name: Theorem LP, San Mateo, California 94402, USA
    index: 2
  - name: Jet Propulsion Laboratory, California Institute of Technology, Pasadena, CA 91109, USA
    index: 5
  - name: Department of Physics and Center for Computational Research, East Hall, University of Rhode Island, Kingston, RI 02881
    index: 6
  - name: Theoretical Astrophysics 350-17, California Institute of Technology, Pasadena, California 91125, USA
    index: 7
date: 3 April 2024
bibliography: paper.bib


---

# Summary


Gravitational waves are ripples in space-time caused by the motion of massive objects. One of the most astrophysically important sources of gravitational radiation is caused by two orbiting compact objects, such as black holes and neutron stars, that slowly inspiral and merge. The motion of these massive objects generates gravitational waves that radiate to the far field where gravitational-wave detectors can observe them. Complicated partial or ordinary differential equations govern the entire process. 

Traditionally, the dynamics of compact binary systems and the emitted gravitational waves have been computed by expensive simulation codes that can take days to months to run. A key simulation output is the gravitational wave signal for a particular set of parameter values describing the system, such as the black holes' masses and spins. The computed signal is required for a diverse range of multiple-query applications, such as template bank generation for searches, parameter estimation, mock data analysis, studies of model bias, and tests of general relativity, to name a few. In such settings, the high-fidelity signal computed from differential equations is often too slow to be directly used.

![Example gravitational wave prediction from a surrogate model compared with numerical relativity for a precessing binary black hole system. This particular numerical relativity simulation took 70,881 CPU-hours (about 1.75 months using 56 cores on the supercomputer Frontera), while the surrogate model can be evaluated in about 100 milliseconds. \label{fig:gws}](gwsurrogate.png)

Surrogate models offer a practical way to dramatically accelerate model evaluation while retaining the high-fidelity accuracy of the expensive simulation code; an example is shown in Fig.\ref{fig:gws}. Surrogate models can be constructed in various ways, but what separates these models from other modeling frameworks is that they are primarily data-driven. Given a training set of gravitational waveform data sampling the parameter space, a model is built by following three steps:

1. Feature extraction: the waveform is decomposed into *data pieces* that are simple to model,
2. Dimensionality reduction: each data piece is approximated by a low-dimensional vector space, which reduces the degrees of freedom we need to model, and 
3. Regression: fitting and regression techniques are applied to the low-dimensional representation of each data piece over the parameter space defining the model. 

These model-building steps result in an HDF5 file defining the surrogate model's data and structure, which is stored on Zenodo. The GWSurrogate package provides access to these HDF5 files through its catalog interface, and all available models and their HDF5 files can be found in `gwsurrogate.catalog.list()`. For a recent overview of surrogate modeling as used in gravitational wave astrophysics, please see Section 5 of @LISAConsortiumWaveformWorkingGroup:2023arg.

The development of ``GWSurrogate`` is hosted on [GitHub](https://github.com/sxs-collaboration/gwsurrogate) and distributed through both [PyPI](https://pypi.org/project/gwsurrogate/) and [Conda](https://anaconda.org/conda-forge/gwsurrogate/). Quick start guides are found on the project's [homepage](https://github.com/sxs-collaboration/gwsurrogate) while model-specific documentation is described through a collection of model-specific [Jupyter notebooks](https://github.com/sxs-collaboration/gwsurrogate/tree/master/tutorial). Automated testing is run on [GitHub Actions](https://github.com/sxs-collaboration/gwsurrogate/actions).


# Statement of need

``GWSurrogate`` is a Python package that provides an easy to use interface to gravitational wave surrogate models. More specifically, ``GWSurrogate`` gravitational wave models provide evaluation of
$$
 h_{\tt S}(t, \theta, \phi;\Lambda) = \sum^{\infty}_{\ell=2} \sum_{m=-\ell}^{\ell} h_{\tt S}^{\ell m}(t;\Lambda) ~^{-2}Y_{\ell m}(\theta, \phi) \,,
$$
where $^{-2}Y_{\ell m}$ are the spin$=-2$ weighted spherical harmonics and $\Lambda$ describes the model's parameterization. The surrogate model provides fast evaluations for the modes, $h_{\tt S}^{\ell m}$. As described more fully in the documentation, the high-level API allows users direct access to the modes $\{h_{\tt S}^{\ell m}(t)\}$ (as a Python dictionary) or assembles the sum $h_{\tt S}(t, \theta, \phi)$ at a particular location $(\theta, \phi)$. The models implemented in ``GWSurrogate`` are intended to be used in production data analysis efforts. As such,

- computationally expensive operations (e.g., interpolation onto uniform time grids) are implemented by wrapping low-level C code for speed, whereas ``GWSurrogate`` provides a user-friendly interface to the high-level waveform evaluation API,
- models implemented in ``GWSurrogate`` follow the waveform convention choices of the LIGO-Virgo-Kagra collaboration, thus ensuring that downstream data analysis codes can use ``GWSurrogate`` models without needing to worry about different conventions, and
- ``GWSurrogate`` models can be directly evaluated in either physical units (often used in data analysis studies) and dimensionless units (often used in theoretical studies) where all dimensioned quantities are expressed in terms of the system's total mass.

Currently, there are 15 supported surrogate models (@Blackman:2015pia, @OShaughnessy:2017tak, @Blackman:2017dfb, @Varma:2018mmi, @Yoo:2023spi, @Yoo:2022erv, @Varma:2019csw, @Barkett:2019tus, @Rifat:2019ltp, @Islam:2022laz, @Field:2013cfa). These models vary in their duration, included physical effects (e.g. nonlinear memory, tidal forces, harmonic modes retained, eccentricity, mass ratio extent, precession effects, etc), and underlying solution method (e.g. Effective One Body, numerical relativity, and black hole perturbation theory). Details about all models can be found by doing `gws.catalog.list(verbose=True)`, while the ``GWSurrogate`` [homepage](https://github.com/sxs-collaboration/gwsurrogate) summarizes the state-of-the-art models for each particular problem. Certain models allow for additional functionality such as returning the dynamics of the binary black hole; these special features are described further in model-specific [example notebooks](https://github.com/sxs-collaboration/gwsurrogate/tree/master/tutorial).


# Acknowledgements

We acknowledge our many close collaborators for their contribution to the development of surrogate models. We further acknowledge the community of ``GWSurrogate`` users who have contributed pull requests and opened issues, including Mike Boyle, Collin Capano, Raffi Enficiaud, Oliver Jennrich, Duncan Macleod, Alex Nitz, Seth Olsen, Swati Singh, and Avi Vajpeyi. ``GWSurrogate`` has been developed over the past 10 years with continued support from the National Science Foundation, most recently through NSF grants PHY-2110496, PHY-2309301, and DMS-2309609.

# References