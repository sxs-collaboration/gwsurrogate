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
  EOB = gws.EvaluateSurrogate('gwsurrogate/tutorial/EOBNRv2_example/EOBNRv2_q1_2_NoSpin_SingleModes/l2_m2_len12239M_SurID19poly/')
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

__copyright__ = "Copyright (C) 2014 Scott Field, Chad Galley"
__email__ = "sfield@umassd.edu"
__status__ = "testing"
__author__ = "Jonathan Blackman, Scott Field, Chad Galley"
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from .surrogate import *
from . import catalog
