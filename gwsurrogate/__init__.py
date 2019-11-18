"""GWSurrogate
   ===========

Provides 
  1. An to use easy interface to gravitational wave surrogate models
  2. Up-to-date database of surrogates and tools for downloading them
  3. Helper routines required for basic data analysis tasks


Example usage
-------------

To plot the EOBNRv2 surrogate included with this package (others available for download)

  import gwsurrogate as gw
  EOB = gws.EvaluateSurrogate('EOBNRv2_q1_2_NoSpin_SingleModes/l2_m2_len12239M_SurID19poly/')
  EOB.plot_sur(q_eval = 1.3)

Additional examples can be found in the accompanying ipython notebooks.


Surrogate data format
---------------------

Both HDF5 and text based surrogates assume

1) Gravitational waves are written as: 

        h = h_+ + i h_x = Ae^{i phi} 

2) The following surrogate data is available:

    SurrogateID (string, repeat parent directory name)
    B (complex matrix, basis-by-times)
    eim_indices (array of integers, labeled from 0)
    greedy_points (ordered by their selection)
    tmin, tmax, dt. (Text surrogates store these in time_info.txt)
    qmin_fit, qmax_fit (range used for fitting).
    affine_map (Boolean; whether affine map to reference interval was
                used; use as flag for evaluating fits)
    fit_coeff_amp
    fit_coeff_phase (coefficients of fitting functions; eim_indices-by-coefficients)
    V (not its inverse; to reconstruct the the orthonormal reduced basis E = B V)
    R (matrix of coefficients relating basis and waveforms, H = E R)

3) Each surrogate model is defined by a folder name (text-based surrogates)

        MODELNAME_q[QMIN]_[QMAX]_NoSpin_[Multi(Single)]Mode/l[ELL]_m[M]_len[TIME]M_SurID[RBDIM]poly

   where, for example, MODELNAME = "EOBNRv2" and []-quantities are determined by the surrogate's setting

or file name (hdf5-based surrogates)

        MODELNAME.h5

4) metadata.txt or info.data contains a detailed description of the surrogate model


surrogate.py
------------

Defines the classes HDF5Surrogate and TextSurrogate. These are low-level 
classes for loading, evaluating and plotting surrogate models (stored 
as text or hdf5 data files) as they are exported from the surrogate building 
code; these surrogates are limited to a fixed sampling rate and are dimensionless.

The EvaluateSurrogate class can be used to generate astrophysical surrogates
depending on the masses, distance to the source another other parameters 
of interest.

"""

from . import surrogate
__author__ = surrogate.__author__
__email__ = surrogate.__email__
__copyright__ = surrogate.__copyright__
__license__ = surrogate.__license__
__version__ = surrogate.__version__
__doc__ = surrogate.__doc__

from .surrogate import *
from . import catalog
from . import spline_interp_Cwrapper
from . import precessing_utils
