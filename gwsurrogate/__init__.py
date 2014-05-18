"""gwsurrogate
   ===========

Provides 
  1. An easy interface to gravitational wave surrogate models
  2. Up-to-date database of surrogates and tools for downloading them
  3. Helper routines required for basic data analysis tasks


Example usage
-------------

To plot an EOBNRv2 surrogate valid for mass ratios between 1 and 2, and of 
length about 12,000M

  ipmort gwsurrogate as gw
  EOB = gw.TextSurrogate('EOBNRv2/EOBNRv2_q1_2_NoSpin_SingleModes/l3_m3_len12241M_SurID15poly/')
  EOB.plot(1.2)

(Additional examples for text and hdf5 based surrogates can be 
found in the accompanying ipython notebooks.)


Surrogate data format
---------------------

Both HDF5 and text based surruogates assume 

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

3) Each surrogate model is defined by a folder name (text) 

        MODELNAME_q[QMIN]_[QMAX]_NoSpin_[Multi(Single)]Mode/l[ELL]_m[M]_len[TIME]M_SurID[RBDIM]poly

or file name (hdf5)  

        ADD ME HERE!!!!

where, for example, MODELNAME = "EOBNRv2" and []-quantities are determined by the surrogate's setting

4) metadata.txt or info.data contains a detailed description of the surrogate model


surrogate.py
------------

Defines the classes HDF5Surrogate and TextSurrogate. These are 
low-level classes for loading, evaluating and plotting 
surrogate models (stored as text or hdf5 data files) 
as they are exported from the surrogate building code. 
As such, these surrogates are limited to a fixed sampling rate. 
Evaluations are provided for by specifcying any value of mass 
ratio q which lies within the range for which the surrogate is valid. 


==== NOTES BELOW THIS LINE ARE OUTDATED, KEEP FOR NOW =======


LALsurrogate.py
---------------
Defines the class LALSurrogateEOB. Similar to SurrogateGW excpet that 
surrogates can be generated from interpolants for waveforms with different 
total masses while sampled at the same rate. [NOTE: Unnormalized amplitude 
dependence on parameters is not yet implemented.] To use:


1) Import the routines from "import EvaluateSurrogate_v2" or "from EvaluateSurrogate_v2 import *"

To interpolate the matrix of reduced basis functions:

1) Create instance of InterpolateB class. Must pass location of data files upon initialization, e.g., "interp = InterpolateB('SurrogateQ1to2/')"

2) Generate and save basis function interpolants to hdf5 format with "interp.save_interp('SurrogateQ1to2/')". Options include compressQ, which if True will compress the data using gzip and if False will do no compression. Interpolants will be stored in 'SurrogateQ1to2/B_interp.hdf5'.


To generate an EOB surrogate waveform for a given total mass Mtot and mass ratio q:

1) Create instance of LALSurrogateEOB class. Must pass location of data files upon initialization, e.g., "eob = LALSurrogateEOB('SurrogateQ1to2/')"

2) Generate EOB waveforms by specifying a q and Mtot value (in solar masses), e.g., "t, hp, hc = eob(1.2, 60.)"

3) Plot the same EOB waveform with, e.g., "eob.plot_surrogate(1.2, 60.)"

4) Plot multiple surrogate waveforms with, e.g., "eob.plot_surrogates([[1.2, 60.], [1.2, 80.], [1.2, 100.]])". Options "hpQ" and "hcQ" allow for the plus and/or cross polarizations to be plotted. Default values are "hpQ=True", "hcQ=False".

"""

__copyright__ = "Copyright (C) 2014 GW surrogate group"
__email__ = "sfield@umd.edu"
__status__ = "testing"
__author__ = "Scott Field, Chad Galley"

__license__ = """
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see U{http://www.gnu.org/licenses/}.
"""

from surrogate import *


