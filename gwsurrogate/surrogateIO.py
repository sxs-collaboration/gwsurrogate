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



##############################################
class TextSurrogateIO:
  """base class for single-mode text-based surrogate format. this class does nothing, inherit from it"""

  ### Files which define a text-based surrogate (both read and write) ###

  #variable_name_file   = file_name
  _time_info_file       = 'time_info.txt' # either tuple (ti,tf,dt) or Nx2 matrix for N times and weights
  _fit_interval_file    = 'param_fit_interval.txt'
  _parameterization     = 'parameterization.txt'
  _greedy_points_file   = 'greedy_points.txt'
  _eim_indices_file     = 'eim_indices.txt'
  _affine_map_file      = 'affine_map.txt'
  _fitparams_phase_file = 'fit_coeff_phase.txt'
  _fitparams_amp_file   = 'fit_coeff_amp.txt'
  _fitparams_norm_file  = 'fit_coeff_norm.txt'
  _fit_type_phase_file  = 'fit_type_phase.txt'
  _fit_type_amp_file    = 'fit_type_amp.txt'
  _fit_type_norm_file   = 'fit_type_norm.txt'
  _B_i_file             = 'B_imag.txt'
  _B_r_file             = 'B_real.txt'
  _V_i_file             = 'V_imag.txt'
  _V_r_file             = 'V_real.txt'
  _R_i_file             = 'R_im.txt'
  _R_r_file             = 'R_re.txt'


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __init__(self, sdir):
    pass

##############################################
class TextSurrogateRead(TextSurrogateIO):
  """Load single-mode, text-based surrogate"""

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __init__(self, sdir):
    """initialize single-mode surrogate defined by text files located in directory sdir"""

    ### sdir is defined to the the surrogate's ID ###
    self.SurrogateID = sdir

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

    ### greedy points (ordered by RB selection) ###
    self.greedy_points = np.loadtxt(sdir+self._greedy_points_file)

    ### empirical time index (ordered by EIM selection) ###
    self.eim_indices = np.loadtxt(sdir+self._eim_indices_file,dtype=int)

    ### Complex B coefficients ###
    B_i    = np.loadtxt(sdir+self._B_i_file)
    B_r    = np.loadtxt(sdir+self._B_r_file)
    self.B = B_r + (1j)*B_i

    ### Deduce sizes from B ###
    self.dim_rb       = B_r.shape[1]
    self.time_samples = B_r.shape[0]

    ### Information about phase/amp/norm parametric fits ###
    self.fitparams_phase = np.loadtxt(sdir+self._fitparams_phase_file)
    self.fitparams_amp   = np.loadtxt(sdir+self._fitparams_amp_file)
    self.fitparams_norm  = np.loadtxt(sdir+self._fitparams_norm_file)
    self.affine_map      = bool(np.loadtxt(sdir+self._affine_map_file))
    self.fit_type_phase  = self.get_string_key(sdir+self._fit_type_phase_file)
    self.fit_type_amp    = self.get_string_key(sdir+self._fit_type_amp_file)
    self.fit_type_norm   = self.get_string_key(sdir+self._fit_type_norm_file)

    self.norm_fit_func  = my_funcs[self.fit_type_norm]
    self.phase_fit_func = my_funcs[self.fit_type_phase]
    self.amp_fit_func   = my_funcs[self.fit_type_amp]

    ### Information about surrogate's parameterization ###
    self.parameterization = self.get_string_key(sdir+self._parameterization)
    self.get_surr_params  = my_funcs[self.parameterization]
		
    # TODO: set to false and don't load data if norm information not provided
    self.norms = True

    ### Vandermonde V such that E (orthogonal basis) is E = BV ###
    V_i    = np.loadtxt(sdir+self._V_i_file)
    V_r    = np.loadtxt(sdir+self._V_r_file)
    self.V = V_r + (1j)*V_i

    ### R matrix such that waveform basis H = ER ###
    R_i    = np.loadtxt(sdir+self._R_i_file)
    R_r    = np.loadtxt(sdir+self._R_r_file)
    self.R = R_r + (1j)*R_i

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def get_string_key(self,fname):
    """ return a single word string from file """

    fp = open(fname,'r')
    keyword = fp.readline()
    return keyword[0:-1]


##############################################
class TextSurrogateWrite(TextSurrogateIO):
  """Export single-mode, text-based surrogate"""

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
    """ Write surrogate data (text) in standard format.
		
      Input:
      ======
      time_info       -- tuple (dt, tmin, tmax)
      B               -- empirical interpolant operator (`B matrix`)
      eim_indices     -- indices of empirical nodes from time series array `t`
      greedy_points   -- parameters selected by reduced basis greedy algorithm
      V               -- Generalized Vandermonde matrix from empirical 
                         interpolation method
      R               -- matrix coefficients relating the reduced basis to the 
                         selected waveforms
      fit_interval    -- min/max values of parameters used for surrogate fitting
      affine_map      -- mapped parameter domain to reference interval for fitting? 
                        (True/False)
      fitparams_amp   -- fitting parameters for waveform amplitude
      fitparams_phase -- fitting parameters for waveform phase
      fitparams_norm  -- fitting parameters for waveform norm
      fit_type_phase  -- key to select parametric fitting function (phase)
      fit_type_amp    -- key to select parametric fitting function (amp)
      fit_type_norm   -- key to select parametric fitting function (norm) 
      parameterization-- key to select map from q,M to surrogate's parameterization"""

    # TODO: flag to zip folder with tar -cvzf SURROGATE_NAME.tar.gz SURROGATE_NAME/

    ### pack mass ratio interval (for fits) and time info ###
    # TODO: should save full time series if necessary
    self.np_savetxt_safe(self.SurrogateID+self._fit_interval_file,fit_interval)
    self.np_savetxt_safe(self.SurrogateID+self._time_info_file,time_info)
    self.np_savetxt_safe(self.SurrogateID+self._greedy_points_file,greedy_points,fmt='%2.16f')
    self.np_savetxt_safe(self.SurrogateID+self._eim_indices_file,eim_indices,fmt='%i')
    self.np_savetxt_safe(self.SurrogateID+self._B_i_file,B.imag)
    self.np_savetxt_safe(self.SurrogateID+self._B_r_file,B.real)
    self.np_savetxt_safe(self.SurrogateID+self._fitparams_phase_file,fitparams_phase)
    self.np_savetxt_safe(self.SurrogateID+self._fitparams_amp_file,fitparams_amp)
    self.np_savetxt_safe(self.SurrogateID+self._affine_map_file,np.array([int(affine_map)]),fmt='%i')
    self.np_savetxt_safe(self.SurrogateID+self._V_i_file,V.imag)
    self.np_savetxt_safe(self.SurrogateID+self._V_r_file,V.real)
    self.np_savetxt_safe(self.SurrogateID+self._R_i_file,R.imag)
    self.np_savetxt_safe(self.SurrogateID+self._R_r_file,R.real)
    self.np_savetxt_safe(self.SurrogateID+self._fitparams_norm_file,fitparams_norm)
    self.np_savetxt_safe(self.SurrogateID+self._fit_type_phase_file,[fit_type_phase],'%s')
    self.np_savetxt_safe(self.SurrogateID+self._fit_type_amp_file,[fit_type_amp],'%s')
    self.np_savetxt_safe(self.SurrogateID+self._fit_type_norm_file,[fit_type_norm],'%s')
    self.np_savetxt_safe(self.SurrogateID+self._parameterization,[parameterization],'%s')

