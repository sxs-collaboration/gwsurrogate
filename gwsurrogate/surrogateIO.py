""" Basic IO functionality for surrogates """

from __future__ import division

__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@astro.cornell.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Scott Field, Chad Galley"

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

import numpy as np
import os as os
from parametric_funcs import function_dict as my_funcs


surrogate_description = """* Description of tags:
     
    These files control the relationship between data and surrogate evaluations

      surrogate_mode_type -- key to control the relationship between the basis, 
                             given by B below, and the temporal degrees of freedom,
                             given by evaluating parametric fits.
      fit_type_phase      -- key to select parametric fitting function (phase)
      fit_type_amp        -- key to select parametric fitting function (amp)
      parameterization    -- key to select map from q,M to surrogate's parameterization
      affine_map          -- mapped parameter domain to reference interval for fitting? 
                             supported maps: [low,high] -> [-1,1], [0,1], and [low,high]
      fit_type_norm       -- (Optional) key to select parametric fitting function (norm)

* Description of data:

    Surrogate evaluations are carried out by various operations/evaluations on
    these data files

      time_info       -- (i) tuple (dt, tmin, tmax) OR (ii) Nx2 matrix with 
                         times and quadrature weights
      fit_interval    -- min/max values of parameters used for surrogate fitting
      fitparams_amp   -- fitting parameters for waveform amplitude
      fitparams_phase -- fitting parameters for waveform phase
      greedy_points   -- parameters selected by reduced basis greedy algorithm
      B_1/B_2         -- Basis matricies (their meaning is described below)
      eim_indices     -- (Optional) indices of empirical nodes from time series array `t`
      fitparams_norm  -- (Optional) fitting parameters for waveform norm
      V_1/V_2         -- (Optional) Generalized Vandermonde matrix from empirical 
                         interpolation method
      R_1/R_2         -- (Optional) matrix coefficients relating the reduced 
                         basis to the selected waveforms



* Surrogate data's dependency on surrogate_mode_type:

   Inspecting the _h_sur member function of EvaluateSingleModeSurrogate 
   shows how the surrogate type is used. Briefly,

    (1) 'waveform_basis' -- The columns of B span a linear approximation space
                            for the complex waveform mode such that h ~ np.dot(self.B, h_EIM)
                            In this case, B_1 (B_2) are the real (imaginary) parts
                            of the basis matrix B. eim_indices is a vector of numbers
                            whose length is exactly the number of columns of B.
                            B = B_1 + j*B_2 is the empirical interpolant operator (`B matrix`)


    (2) 'amp_phase_basis' -- The columns of B.real (B.imag) span a linear approximation
                             space for the amplitude (phase) such that 
                             h ~ np.dot(self.B.real, A_EIM) * exp(1j*np.dot(self.B.imag, Phase_EIM) ).
                             In this case, B_1 (B_2) are the amplitude (phase) basis. 
                             eim_indices contains 2 vectors, the first vector's length
                             matches B_1's columns and the second vector's length
                             matches B_2's columns."""


##############################################
class TextSurrogateIO:
  """
* Summary (IO base class): 

    Base class for single-mode text-based surrogate format. This class
    organizes a common set of mandatory and optional data files and
    bookkeeping tags. It serves as a base class from which surrogate read
    and write classes will inherit.


"""
  __doc__+=surrogate_description

  # format variable_name_file   = file_name
  _surrogate_mode_type_file  = 'surrogate_mode_type.txt'
  _parameterization          = 'parameterization.txt'
  _affine_map_file           = 'affine_map.txt'
  _fit_type_phase_file       = 'fit_type_phase.txt'
  _fit_type_amp_file         = 'fit_type_amp.txt'
  _fit_type_norm_file        = 'fit_type_norm.txt'

  # mandatory data files # 
  _time_info_file            = 'time_info.txt'
  _fit_interval_file         = 'param_fit_interval.txt'
  _fitparams_phase_file      = 'fit_coeff_phase.txt'
  _fitparams_amp_file        = 'fit_coeff_amp.txt'
  _B_1_file                  = 'B_1.txt'
  _B_2_file                  = 'B_2.txt'

  # optional data files #
  _greedy_points_file   = 'greedy_points.txt'
  _fitparams_norm_file  = 'fit_coeff_norm.txt'
  _eim_indices_file     = 'eim_indices.txt'
  _V_1_file             = 'V_1.txt'
  _V_2_file             = 'V_2.txt'
  _R_1_file             = 'R_1.txt'
  _R_2_file             = 'R_2.txt'


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __init__(self, sdir):
    pass

##############################################
class TextSurrogateRead(TextSurrogateIO):
  """Load single-mode, text-based surrogate

"""
  __doc__+=surrogate_description

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __init__(self, sdir):
    """initialize single-mode surrogate defined by text files located in directory sdir"""

    surrogate_load_info = '' # add to string, display after loading

    ### sdir is defined to the the surrogate's ID ###
    self.SurrogateID = sdir

    ### type of surrogate (for this harmonic mode) ###
    self.surrogate_mode_type = self.get_string_key(sdir+self._surrogate_mode_type_file)

    ### Surrogate's sampling rate and mass ratio (for fits) ###
    self.time_info    = np.loadtxt(sdir+self._time_info_file)
    self.fit_interval = np.loadtxt(sdir+self._fit_interval_file)
    #self.Mtot        = np.loadtxt(sdir+'Mtot.txt')

    ### unpack time info ###
    if(self.time_info.size == 3):
      self.dt      = self.time_info[2]
      self.tmin    = self.time_info[0]
      self.tmax    = self.time_info[1]

      # Time samples associated with the original data used to build the surrogate
      self.times              = np.arange(self.tmin, self.tmax+self.dt, self.dt)
      # TODO: these are not the weights were the basis are ortho (see basis.ipynb)
      self.quadrature_weights = self.dt * np.ones(self.times.shape)
		
    else:
      self.times              = time_info[:,0]
      self.quadrature_weights = time_info[:,1]

    self.t_units = 'TOverMtot' # TODO: pass this through time_info for flexibility

    ### Complex B coefficients ###
    B_1    = np.loadtxt(sdir+self._B_1_file)
    B_2    = np.loadtxt(sdir+self._B_2_file)

    # TODO: B needs to be handled better 
    if self.surrogate_mode_type  == 'amp_phase_basis':
      self.B_1 = B_1
      self.B_2 = B_2
    elif self.surrogate_mode_type  == 'waveform_basis':
      self.B = B_1 + (1j)*B_2
    else:
      raise ValueError('invalid surrogate type')

    ### Deduce sizes from B ###
    # TODO: dim_rb needs to account for different amp/phase dimensions
    self.dim_rb       = B_1.shape[1]
    self.time_samples = B_1.shape[0]

    ### Information about phase/amp parametric fits ###
    self.fitparams_phase = np.loadtxt(sdir+self._fitparams_phase_file)
    self.fitparams_amp   = np.loadtxt(sdir+self._fitparams_amp_file)

    #self.affine_map     = bool(np.loadtxt(sdir+self._affine_map_file))
    self.affine_map      = self.get_string_key(sdir+self._affine_map_file)

    self.fit_type_phase  = self.get_string_key(sdir+self._fit_type_phase_file)
    self.fit_type_amp    = self.get_string_key(sdir+self._fit_type_amp_file)

    self.phase_fit_func = my_funcs[self.fit_type_phase]
    self.amp_fit_func   = my_funcs[self.fit_type_amp]

    ### Information about surrogate's parameterization ###
    self.parameterization = self.get_string_key(sdir+self._parameterization)
    self.get_surr_params  = my_funcs[self.parameterization]


    ### Load optional parameters if they exist ###

    ### Vandermonde V such that E (orthogonal basis) is E = BV ###
    try:
      V_1    = np.loadtxt(sdir+self._V_1_file)
      V_2    = np.loadtxt(sdir+self._V_2_file)
      self.V = V_1 + (1j)*V_2
    except IOError:
      surrogate_load_info +='Vandermonde not found, '
      self.V = False


    ### greedy points (ordered by RB selection) ###
    try:
      self.greedy_points = np.loadtxt(sdir+self._greedy_points_file)
    except IOError:
      surrogate_load_info += 'Greedy points not found, '
      self.greedy_points = False

    ### R matrix such that waveform basis H = ER ###
    try:
      R_1    = np.loadtxt(sdir+self._R_1_file)
      R_2    = np.loadtxt(sdir+self._R_2_file)
      self.R = R_1 + (1j)*R_2
    except IOError:
      surrogate_load_info += 'R matrix not found, '
      self.R = False

    try: 
      self.fitparams_norm = np.loadtxt(sdir+self._fitparams_norm_file)
      self.fit_type_norm  = self.get_string_key(sdir+self._fit_type_norm_file)
      self.norm_fit_func  = my_funcs[self.fit_type_norm]
      self.norms = True
    except IOError:
      surrogate_load_info += 'Norm fits not found, '
      self.norms = False

    ### empirical time index (ordered by EIM selection) ###
    try:
      self.eim_indices = np.loadtxt(sdir+self._eim_indices_file,dtype=int)
    except IOError:
      surrogate_load_info += 'EIM indices not found.'
      self.eim_indices = False

    #print surrogate_load_info #Q: should we display this?

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def get_string_key(self,fname):
    """ return a single word string from file """

    fp = open(fname,'r')
    keyword = fp.readline()
    if '\n' in keyword:
      return keyword[0:-1]
    else:
      return keyword


##############################################
class TextSurrogateWrite(TextSurrogateIO):
  """Export single-mode, text-based surrogate

"""
  __doc__+=surrogate_description

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __init__(self, sdir):
    """open single-mode surrogate, to be located in directory sdir, for writing"""

    if( not(sdir[-1:] is '/') ):
      raise Exception, "path name should end in /"
    try:
      os.mkdir(sdir)
      print "Successfully created a surrogate directory...use write_text to export your surrogate!"
    except OSError:
      print "Could not create a surrogate directory. Not ready to export, please try again."

    ### sdir is defined to the the surrogate's ID ###
    self.SurrogateID = sdir


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def np_savetxt_safe(self,fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# '):
    """ numpys savetext without overwrites """

    if os.path.isfile(fname):
      raise Exception, "file already exists"
    else: 
      np.savetxt(fname,X,fmt=fmt)


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def write_text(self, time_info, B, eim_indices, greedy_points, fit_interval, affine_map, \
                 fitparams_amp, fitparams_phase, fitparams_norm, V, R,fit_type_phase,\
                 fit_type_amp, fit_type_norm,parameterization):
    """ Write surrogate data (text) in standard format."""
		

    # TODO: flag to zip folder with tar -cvzf SURROGATE_NAME.tar.gz SURROGATE_NAME/

    ### pack mass ratio interval (for fits) and time info ###
    # TODO: should save full time series if necessary
    self.np_savetxt_safe(self.SurrogateID+self._fit_interval_file,fit_interval)
    self.np_savetxt_safe(self.SurrogateID+self._time_info_file,time_info)
    self.np_savetxt_safe(self.SurrogateID+self._greedy_points_file,greedy_points,fmt='%2.16f')
    self.np_savetxt_safe(self.SurrogateID+self._eim_indices_file,eim_indices,fmt='%i')
    self.np_savetxt_safe(self.SurrogateID+self._B_1_file,B.real)
    self.np_savetxt_safe(self.SurrogateID+self._B_2_file,B.imag)
    self.np_savetxt_safe(self.SurrogateID+self._fitparams_phase_file,fitparams_phase)
    self.np_savetxt_safe(self.SurrogateID+self._fitparams_amp_file,fitparams_amp)
    self.np_savetxt_safe(self.SurrogateID+self._affine_map_file,np.array([int(affine_map)]),fmt='%i')
    self.np_savetxt_safe(self.SurrogateID+self._V_1_file,V.real)
    self.np_savetxt_safe(self.SurrogateID+self._V_2_file,V.imag)
    self.np_savetxt_safe(self.SurrogateID+self._R_1_file,R.real)
    self.np_savetxt_safe(self.SurrogateID+self._R_2_file,R.imag)
    self.np_savetxt_safe(self.SurrogateID+self._fitparams_norm_file,fitparams_norm)
    self.np_savetxt_safe(self.SurrogateID+self._fit_type_phase_file,[fit_type_phase],'%s')
    self.np_savetxt_safe(self.SurrogateID+self._fit_type_amp_file,[fit_type_amp],'%s')
    self.np_savetxt_safe(self.SurrogateID+self._fit_type_norm_file,[fit_type_norm],'%s')
    self.np_savetxt_safe(self.SurrogateID+self._parameterization,[parameterization],'%s')

