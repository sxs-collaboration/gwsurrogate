""" Basic IO functionality for surrogates """

from __future__ import division  # for python 2

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
import h5py
from .parametric_funcs import function_dict as my_funcs
from .new.spline_evaluation import TensorSplineGrid, fast_tensor_spline_eval
import collections

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
      t_units         -- surrogate's temporal units (values = 'TOverMtot', ...)
      surrogate_units -- currently hardcoded to be 'dimensionless'
                         NOTE: error raised if t_units not 'TOverMtot'
                               since surrogate_units and t_units describe the same
                               information as evaluations are currently coded.
                               In principle they *could* be different
      fit_interval    -- min/max values of parameters used for surrogate fitting
      fitparams_amp   -- fitting parameters for waveform amplitude
      fitparams_phase -- fitting parameters for waveform phase
      greedy_points   -- parameters selected by reduced basis greedy algorithm
      B_1/B_2         -- Basis matrices (their meaning is described below)
      eim_indices     -- (Optional) indices of empirical nodes from time series array `t`
      fitparams_norm  -- (Optional) fitting parameters for waveform norm
      V_1/V_2         -- (Optional) Generalized Vandermonde matrix from empirical 
                         interpolation method
      R_1/R_2         -- (Optional) matrix coefficients relating the reduced 
                         basis to the selected waveforms
      spline_knots    -- (Only for fast_spline surrogates) A concatenated list of spline
                         knots in each dimension
      n_spline_knots  -- (Only for fast_spline surrogates) A list of the number of knots
                         in each dimension


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
class SurrogateBaseIO:
  """
* Summary (IO base class): 

    Base class for single-mode text-based surrogate format. This class
    organizes a common set of mandatory and optional data files and
    bookkeeping tags. It serves as a base class from which surrogate read
    and write classes will inherit.


"""
  __doc__+=surrogate_description

  # format variable_name_{h5,txt} = {file_name,h5_group}
  # Note: some of these fields are optional

  ## REQUIRED DATA ##

  _surrogate_ID_h5          = 'surrogate_ID' # From local folder name for text surrogates 

  _surrogate_mode_type_txt  = 'surrogate_mode_type.txt'
  _surrogate_mode_type_h5   = 'surrogate_mode_type' # .txt

  _parameterization_txt     = 'parameterization.txt'
  _parameterization_h5      = 'parameterization' # .txt

  _time_info_txt            = 'time_info.txt' # with min, max, dt or (times, quad weights). depricated (11/23/2016)
  _tmin_h5                  = 'tmin' # 
  _tmax_h5                  = 'tmax' # Pass tmin, tmax and dt, or... 
  _dt_h5                    = 'dt'   #

  _times_h5                 = 'times'              # pass a grid of times and quadrature weights 
  _quadrature_weights_h5    = 'quadrature_weights' # 

  _fit_interval_txt         = 'param_fit_interval.txt'
  _fit_min_h5               = 'fit_min' #  _fit_interval_txt = 'param_fit_interval.txt' (both max and min)
  _fit_max_h5               = 'fit_max'

  _B_1_txt                  = 'B_1.txt'
  _B_2_txt                  = 'B_2.txt'
  _B_h5                     = 'B' # B_1.txt
  _B_phase_h5               = 'B_phase' # B_2.txt


  ## COULD BE OPTIONAL DATA (depends on surrogate type) ###

  _affine_map_txt           = 'affine_map.txt'
  _affine_map_h5            = 'affine_map' # .txt

  _fit_type_phase_txt       = 'fit_type_phase.txt'
  _fit_type_phase_h5        = 'fit_type_phase'   # .txt

  _fit_type_amp_txt         = 'fit_type_amp.txt'
  _fit_type_amp_h5          = 'fit_type_amp' # .txt

  _fit_type_norm_txt        = 'fit_type_norm.txt'
  _fit_type_norm_h5         = 'fit_type_norm' # .txt

  _fitparams_phase_txt      = 'fitparams_phase.txt' # switched from fit_coeff_phase.txt 11/9/2016
  _fitparams_phase_h5       = 'fitparams_phase' #  .txt

  _fitparams_amp_txt        = 'fitparams_amp.txt'   # switched from fit_coeff_amp.txt 11/9/2016
  _fitparams_amp_h5         = 'fitparams_amp'  # .txt

  _fitparams_norm_txt  = 'fitparams_norm.txt' # switched from fit_coeff_norm.txt 11/9/2016
  _fitparams_norm_h5   = 'fitparams_norm' # .txt


  ## OPTIONAL DATA ##
  _greedy_points_txt   = 'greedy_points.txt'
  _greedy_points_h5    = 'greedy_points' # .txt

  _eim_indices_txt       = 'eim_indices.txt'
  _eim_indices_h5        = 'eim_indices'       # .txt... text: eim_indices contains 2 vectors if amp/phase
  _eim_indices_phase_h5  = 'eim_indices_phase' # rolled into eim_indices
  _eim_amp_h5            = 'eim_amp' # text analog? only used to plot 
  _eim_phase_h5          = 'eim_phase' # text analog? only used to plot

  _V_1_txt             = 'V_1.txt'
  _V_2_txt             = 'V_2.txt'
  _V_h5                = 'V' # V_1.txt V_2.txt

  _R_1_txt             = 'R_1.txt'
  _R_2_txt             = 'R_2.txt'
  _R_h5                = 'R' # R_1.txt R_2.txt

  _t_units_txt         = 't_units.txt'
  _t_units_h5          = 't_units' # .txt



  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __init__(self):

    ### Make list of required data for reading/writing surrogate data ###
    # this is the minimum data. Surrogate evaluations may require more than
    # the minimum
    self.required = ['times', 'B', 'fitparams_amp', 'fitparams_phase',
                     'fit_min', 'fit_max', 'fit_type_amp', 'fit_type_phase',
                     'surrogate_mode_type', 'parameterization']


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def print_required(self):
    """ Print variable names required for importing and exporting surrogate data"""
    
    print("\nGWSurrogate requires data for the following:")
    
    for kk in self.required:
      print("\t"+kk)
    
    pass

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def check_required(self, dict):
    """ Check if input dictionary has the minimum required surrogate data"""
    #keys = dict.keys()
    
    for kk in self.required:
      #if kk not in keys:
      if kk not in dict:
        raise Exception("\nGWSurrogate requires data for "+kk)
    
    return list(dict.keys())



##############################################
class H5Surrogate(SurrogateBaseIO):
  """Load or export a single-mode surrogate in terms of the function's amplitude and phase from HDF5 data format"""

  __doc__ += surrogate_description
  
  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __init__(self, file=None, mode=None, subdir='', closeQ=True):
    SurrogateBaseIO.__init__(self)
    
    ### Get mode name if supplied and check if subdir ends with '/' ###
    if subdir != '':
      if subdir[-1] == '/':
        self.subdir = subdir
        self.mode = subdir[:-1]
      else:
        self.subdir = subdir + '/'
        self.mode = subdir
    else:
      self.subdir = subdir
    
    ### Check file mode if specified ###
    if mode is not None:
      if mode in ['r', 'w', 'r+', 'a']:
        self._mode = mode
      else:
        raise Exception("File mode not recognized. Must be 'r', 'w', 'r+', 'a'.")
    
    ### Check if file is a pointer or path name (string) and open if in 'r' mode ###
    if file is not None:
      self.type = type(file)
      if self.type == str:
        if self._mode == 'r':
          try:
            self.file = h5py.File(file, 'r')
          except:
            pass
        if self._mode == 'w':
          self.file = h5py.File(file, 'w')
      elif self.type == h5py._hl.files.File:
        if self._mode == 'r' or self._mode == 'w':
          self.file = file
    
      ### If mode is 'r' then import surrogate data ###
      if self._mode == 'r':
        self.load_h5(file, subdir=self.subdir, closeQ=closeQ)
      
    pass
  
  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def prepare_mode_data(self, data_mode_class):
    """ Prepare a dictionary to export with entries filled from imported single-mode surrogate data"""
    
    # data_mode_class will usually be of the form surrogate_class.single_mode[<mode_key>]
    
    dict = {}
    
    for kk in data_mode_class.keys:
      dict[kk] = data_mode_class.__dict__[kk]
    
    if 'mode' in data_mode_class.__dict__:
      dict['mode'] = data_mode_class.__dict__['mode']
    
    return dict
  
  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def chars_to_string(self, chars):
    return "".join(chr(cc) for cc in chars)

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def string_to_chars(self, string):
    return [ord(cc) for cc in string]
  
  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def prepare_data(self, data_class):
    """ Prepare a dictionary to export with entries filled from imported surrogate data"""
    return [self.prepare_mode_data(data_class.single_modes[mm]) for mm in data_class.modes]
      
  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def load_h5(self, file, subdir='', closeQ=True):
    
    ### Open file based on input being a filename or file pointer ###
    self.type = type(file)
    if self.type == str:
      try:
        self.file = h5py.File(file, 'r')
      except:
        raise Exception("Cannot open file.")
    elif self.type == h5py._hl.files.File:
      self.file = file
    
    ### Get data keys listing all available surrogate data ###
    if subdir == '':
      self.keys = list(self.file.keys())
    else:
      # Get keys in the given subdirectory
      self.keys = list(self.file[subdir[:-1]].keys())
      
    ### Get SurrogateID ####
    name = self.file.filename.split('/')[-1].split('.')[0]
    if self._surrogate_ID_h5 in self.keys:
      self.surrogate_ID = self.chars_to_string(self.file[subdir+self._surrogate_ID_h5][()])
      if self.surrogate_ID != name:
        print("\n>>> Warning: SurrogateID does not have expected name.")
    else:
      "\n>>> Warning: No surrogate ID found."
    
    ### Get type of basis used to build surrogate 
    # (e.g., basis for complex waveform or for amplitude and phase)
    self.surrogate_mode_type = self.chars_to_string(self.file[subdir+self._surrogate_mode_type_h5][()])
    
    ### Unpack time info ###
    #self.tmin = self.file[subdir+self._tmin_h5][()]
    #self.tmax = self.file[subdir+self._tmax_h5][()]
    #if self._dt_h5 in self.keys:
    #  self.dt = self.file[subdir+self._dt_h5][()]
    #  self.times = np.arange(self.tmin, self.tmax+self.dt, self.dt)
    #  self.quadrature_weights = self.dt * np.ones(self.times.shape)

    if self._dt_h5 in self.keys and self._tmin_h5 in self.keys:
      print(">>> tmin, tmax, dt are depricated as of 11/23/2016.")
      self.tmin = self.file[subdir+self._tmin_h5][()]
      self.tmax = self.file[subdir+self._tmax_h5][()]
      self.dt = self.file[subdir+self._dt_h5][()]
      self.times = np.arange(self.tmin, self.tmax+self.dt, self.dt)
      self.quadrature_weights = self.dt * np.ones(self.times.shape)
    else:
      self.times = self.file[subdir+self._times_h5][:]
      self.tmin  = self.times[0]
      self.tmax  = self.times[-1]
      try:
        self.quadrature_weights = self.file[subdir+self._quadrature_weights_h5][:]
      except:
        self.quadrature_weights = (self.times[1] - self.times[0]) * np.ones(self.times.shape)
        print("\n>>> Warning: Guessing quadrature weights to be identical with %f"%self.quadrature_weights[0])

    #if self._times_h5 not in self.__dict__.keys():
    #  print "\n>>> Warning: No time samples found or generated."
    #if self._quadrature_weights_h5 not in self.__dict__.keys():
    #  print "\n>>> Warning: No quadrature weights found or generated."
    
    if self._t_units_h5 in self.keys:
      self.t_units = self.chars_to_string(self.file[subdir+self._t_units_h5][()])
    else:
      self.t_units = 'TOverMtot'

    ### redundently fill this variable too -- TODO: should have one var only ###
    if(self.t_units == 'TOverMtot'):
      self.surrogate_units = 'dimensionless'
    else:
      raise ValueError('surrogates must be dimensionless')

    ### Greedy points (ordered by RB selection) ###
    if self._greedy_points_h5 in self.keys:
      self.greedy_points = self.file[subdir+self._greedy_points_h5][:]
    else:
      self.greedy_points = None
      print("Cannot load greedy points...OK")
    
    ### Empirical time index (ordered by EIM selection) ###
    if self.surrogate_mode_type == 'amp_phase_basis':
      try:
        self.eim_indices_amp = self.file[subdir+self._eim_indices_h5][:]
        self.eim_indices_phase = self.file[subdir+self._eim_indices_phase_h5][:]
      except KeyError:
        print("Cannot load eim points...OK") 
    elif self.surrogate_mode_type  == 'waveform_basis':
      try:
        self.eim_indices = self.file[subdir+self._eim_indices_h5][:]
      except KeyError:
        print("Cannot load eim points...OK")
    else:
      raise ValueError('invalid surrogate type')

    ### Complex B coefficients ###
    if self.surrogate_mode_type == 'amp_phase_basis':
      self.B_1 = self.file[subdir+self._B_h5][:]
      self.B_2 = self.file[subdir+self._B_phase_h5][:]
    elif self.surrogate_mode_type  == 'waveform_basis':
      self.B = self.file[subdir+self._B_h5][:]	
    else:
      raise ValueError('invalid surrogate type')
    
    ### Information about phase/amp parametric fit ###
    if self._affine_map_h5 in self.keys:
      self.affine_map = self.chars_to_string(self.file[subdir+self._affine_map_h5][()])
    else:
      self.affine_map = 'none'
    self.fitparams_amp = self.file[subdir+self._fitparams_amp_h5][:]
    self.fitparams_phase = self.file[subdir+self._fitparams_phase_h5][:]
    self.fit_min = self.file[subdir+self._fit_min_h5][()]
    self.fit_max = self.file[subdir+self._fit_max_h5][()]
    self.fit_interval = np.array( [self.fit_min, self.fit_max] )
    
    self.fit_type_amp = self.chars_to_string(self.file[subdir+self._fit_type_amp_h5][()])
    self.fit_type_phase = self.chars_to_string(self.file[subdir+self._fit_type_phase_h5][()])

    # TODO: node fitting functions need to be generalized to their own class by using gws.new
    if self.fit_type_amp == "fast_spline_real" and self.fit_type_phase == "fast_spline_imag":
      print("Special case: using fast tensor spline for real and imaginary parts instead of amp/phase")
      print("Loading fast tensor spline breakpoints")

      # TODO: promote data fields (e.g. splint_knots) to SurrogateBaseIO data -- but better to use gws.new
      try: # TODO: 1d surrogates should include n_spline_knots in their data
        n_spline_knots = self.file[subdir+'n_spline_knots'][:]
        remaining_spline_knots = self.file[subdir+'spline_knots'][:]
        spline_knots = []
        for n in n_spline_knots:
          spline_knots.append(remaining_spline_knots[:n])
          remaining_spline_knots = remaining_spline_knots[n:]
      except KeyError: # if n_spline_knots does not exist a KeyError is raised. Old 1d surrogate assumed
        spline_knots = [self.file[subdir+'spline_knots'][:]]

      # setup the function which will be used to evaluate splines
      # TODO: unfortunately, this creates a new grid for each mode.
      #       which does not take advantage of grids that are identical
      #       for different modes.
      #       This will also be slow: each mode will need to find the fast spline data for all modes 
      #       without reusing already computed data
      self.ts_grid = TensorSplineGrid(spline_knots)

      # NOTE: this evaluates the real part -- but its called amp to keep with naming convention -- terrible! (TODO)
      def amp_fit_func(coeffs,xvec):
        amp_eval = fast_tensor_spline_eval(xvec,self.ts_grid,coeffs)
        return amp_eval

      self.amp_fit_func = amp_fit_func

      # NOTE: this evaluates the imaginar part -- but its called phase to keep with naming convention -- terrible! (TODO)
      def phase_fit_func(coeffs,xvec):
        phase_eval = fast_tensor_spline_eval(xvec,self.ts_grid,coeffs)
        return phase_eval

      self.phase_fit_func = phase_fit_func
      
    else:
      self.amp_fit_func   = my_funcs[self.fit_type_amp]
      self.phase_fit_func = my_funcs[self.fit_type_phase]

    if self._fit_type_norm_h5 in self.keys:
      self.fitparams_norm = self.file[subdir+self._fitparams_norm_h5][:]
      self.fit_type_norm = self.chars_to_string(self.file[subdir+self._fit_type_norm_h5][()])
      self.norm_fit_func  = my_funcs[self.fit_type_norm]
      self.norms = True
    else:
      self.norms = False
    
    if self._eim_amp_h5 in self.keys:
      self.eim_amp = self.file[subdir+self._eim_amp_h5][:]
    
    if self._eim_phase_h5 in self.keys:
      self.eim_phase = self.file[subdir+self._eim_phase_h5][:]
    
    ### Transpose matrices if surrogate was built using ROMpy ###
    transposeB = False
    if not self.surrogate_mode_type == 'amp_phase_basis':
      Bshape = np.shape(self.B)
    
      if Bshape[0] < Bshape[1]:
        transposeB = True
        self.B = np.transpose(self.B)
        self.dim_rb = Bshape[0]
        self.time_samples = Bshape[1]
    
      else:
        self.dim_rb = Bshape[1]
        self.time_samples = Bshape[0]
    else: # TODO: elif... to match other similar control statements
        Bshape = np.shape(self.B_1)
        self.dim_rb = Bshape[0]
        self.time_samples = Bshape[1]
        self.dim_rb_phase = np.shape(self.B_2)[0]

    ### Vandermonde V such that E (orthogonal basis) is E = BV ###
    if self._V_h5 in self.keys:
      self.V = self.file[subdir+self._V_h5][:]
      if transposeB:
        self.V = np.transpose(self.V)
    else:
       self.V = None
    
    ### R matrix such that waveform basis H = ER ###
    if self._R_h5 in self.keys:
      self.R = self.file[subdir+self._R_h5][:]
      if transposeB:
        self.R = np.transpose(self.R)
    else:
      self.R = None
        
    ### Information about surrogate's parameterization ###
    self.parameterization = self.chars_to_string(self.file[subdir+self._parameterization_h5][()])
    self.get_surr_params  = my_funcs[self.parameterization]
    
    if closeQ:
      self.file.close()
    
    pass
    
  
  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def write_h5(self, dict, path=None, subdir='', closeQ=True):
    """ Export surrogate data in standard format.
    
    Input:
    ======
      dict -- Dictionary or list of dictionaries 
              containing surrogate data to export.
      
    NOTE: Run print_required() to print the list of 
    the minimum data required by GWSurrogate.
    """
    
    ### Check if path is a filename or pointer and process accordingly ###
    if path is not None:
      if type(path) is str:
        file = h5py.File(path, 'w')
      elif type(path) == h5py._hl.files.File:
        file = path
    else:
      file = self.file
    
    ### Check if dict is for a single mode or multiple modes ###
    if type(dict) is list:
      num_modes = len(dict)
    else:
      num_modes = 1
     
    ### Check that the minimum required surrogate data is given ###
    if num_modes == 1:
      keys = [self.check_required(dict)]
    else:
      keys = [self.check_required(dd) for dd in dict]
    
    ### Export surrogate data to HDF5 file ###
    for ii in range(num_modes):
      
      # Create a group that will house a set of surrogate data for a single mode
      if num_modes == 1:
        data_to_write = dict
        if subdir == '':
          group = file.create_group(data_to_write['mode'])
        else:
          group = file.create_group(subdir)
      else:
        data_to_write = dict[ii]
        group = file.create_group(data_to_write['mode'])
      
      # Write single-mode surrogate data to file, excluding the 'mode' information
      for kk in keys[ii]:

        print("key = %s"%kk)

        if kk not in ['mode', 'surrogate_ID', 'B1_spline_params', 'B2_spline_params']:
            
          dtype = type(data_to_write[kk])
          if dtype is str:
            chars = self.string_to_chars(data_to_write[kk])
            group.create_dataset(kk, data=chars, dtype='int')
          elif dtype is np.ndarray:
            group.create_dataset(kk, data=data_to_write[kk], dtype=data_to_write[kk].dtype, compression='gzip')
          elif isinstance(data_to_write[kk], collections.Callable):
            print("key %s is a function. Not writing to h5 file"%kk)
          elif data_to_write[kk] is None:
            print("key %s has a value of none. Not writing to h5 file"%kk)
          else:
            group.create_dataset(kk, data=data_to_write[kk], dtype=type(data_to_write[kk]))

        elif kk=='surrogate_ID':
          name = file.filename.split('/')[-1].split('.')[0]
          group.create_dataset('surrogate_ID', data=self.string_to_chars(name), dtype='int')
          #group.create_dataset('surrogate_ID', data=self.string_to_chars(data_to_write[kk]), dtype='int')
    
    ### Close file, if requested ###
    if closeQ:
      file.close()
    
    pass


##############################################
class TextSurrogateRead(SurrogateBaseIO):
  """Load single-mode, text-based surrogate"""
  __doc__+=surrogate_description

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __init__(self, sdir):
    """initialize single-mode surrogate defined from text files 
       located in directory sdir. """

    surrogate_load_info = '' # add to string, display after loading

    ### sdir is defined to be the surrogate's ID ###
    self.SurrogateID = sdir

    ### type of surrogate (for harmonic mode) ###
    self.surrogate_mode_type = \
      self.get_string_key(sdir+self._surrogate_mode_type_txt)

    ### Surrogate's sampling rate and mass ratio (for fits) ###
    self.time_info    = np.loadtxt(sdir+self._time_info_txt)
    self.fit_interval = np.loadtxt(sdir+self._fit_interval_txt)
    self.fit_min = self.fit_interval[0]
    self.fit_max = self.fit_interval[1]

    ### unpack time info ###
    # NOTE: models stored in dimensionless T/M, whereas basis may be
    # constructed in T. Basis not necessarily ortho wrt dt (see basis.ipynb)
    # newer surrogate models use basis defined on stored time grid
    if(self.time_info.size == 3):
      self.dt      = self.time_info[2]
      self.tmin    = self.time_info[0]
      self.tmax    = self.time_info[1]

      self.times              = np.arange(self.tmin, self.tmax+self.dt, self.dt)
      self.quadrature_weights = self.dt * np.ones(self.times.shape)
    else:
      self.times              = time_info[:,0]
      self.quadrature_weights = time_info[:,1]
      self.tmin               = np.min(self.times)
      self.tmax               = np.max(self.times)

    self.time_samples = self.times.shape[0]
    self.dt           = self.times[1] - self.times[0]

    try:
      self.t_units = self.get_string_key(sdir+self._t_units_txt)
    except IOError:
      self.t_units = 'TOverMtot'

    ### redundently fill this variable too -- TODO: should have one var only ###
    if(self.t_units == 'TOverMtot'):
      self.surrogate_units = 'dimensionless'
    else:
      raise ValueError('surrogates must be dimensionless')

    ### Complex B coefficients - set ndim=2 in case only 1 basis vector ###
    B_1    = np.loadtxt(sdir+self._B_1_txt,ndmin=2)
    B_2    = np.loadtxt(sdir+self._B_2_txt,ndmin=2)

    ### Consistency check that self.time_samples = B_X.shape[0] ###
    if(self.time_samples != B_1.shape[0] or
       self.time_samples != B_2.shape[0]):
      print(self.time_samples)
      print(B_2.shape[0])
      raise ValueError('temporal and basis dimension mismatch')


    ### Cases handled by in evaluation code ###
    if self.surrogate_mode_type  == 'amp_phase_basis':
      self.B_1 = B_1
      self.B_2 = B_2
      self.B   = None
      self.modeled_data  = 2 # amp, phase data
      self.fits_required = 2 # amp, phase fits
    elif self.surrogate_mode_type  == 'waveform_basis':
      self.B   = B_1 + (1j)*B_2
      self.B_1 = None
      self.B_2 = None
      self.modeled_data  = 1 # complexified waveform data
      self.fits_required = 2 # amp, phase fits
    else:
      raise ValueError('invalid surrogate type')

    ### Information about phase/amp parametric fits ###
    self.fitparams_phase = np.loadtxt(sdir+self._fitparams_phase_txt,ndmin=2)
    self.fitparams_amp   = np.loadtxt(sdir+self._fitparams_amp_txt,ndmin=2)

    self.affine_map      = self.get_string_key(sdir+self._affine_map_txt)

    self.fit_type_phase  = self.get_string_key(sdir+self._fit_type_phase_txt)
    self.fit_type_amp    = self.get_string_key(sdir+self._fit_type_amp_txt)

    self.phase_fit_func = my_funcs[self.fit_type_phase]
    self.amp_fit_func   = my_funcs[self.fit_type_amp]

    ### Information about surrogate's parameterization ###
    self.parameterization = self.get_string_key(sdir+self._parameterization_txt)
    self.get_surr_params  = my_funcs[self.parameterization]


    ### Load optional parameters if they exist ###

    ### Vandermonde V such that E (orthogonal basis) is E = BV ###
    try:
      V_1    = np.loadtxt(sdir+self._V_1_txt)
      V_2    = np.loadtxt(sdir+self._V_2_txt)
      self.V = V_1 + (1j)*V_2
    except IOError:
      surrogate_load_info +='Vandermonde not found, '
      self.V = False


    ### greedy points (ordered by RB selection) ###
    try:
      self.greedy_points = np.loadtxt(sdir+self._greedy_points_txt)
    except IOError:
      surrogate_load_info += 'Greedy points not found, '
      self.greedy_points = False

    ### R matrix such that waveform basis H = ER ###
    try:
      R_1    = np.loadtxt(sdir+self._R_1_txt)
      R_2    = np.loadtxt(sdir+self._R_2_txt)
      self.R = R_1 + (1j)*R_2
    except IOError:
      surrogate_load_info += 'R matrix not found, '
      self.R = False

    try: 
      self.fitparams_norm = np.loadtxt(sdir+self._fitparams_norm_txt)
      self.fit_type_norm  = self.get_string_key(sdir+self._fit_type_norm_txt)
      self.norm_fit_func  = my_funcs[self.fit_type_norm]
      self.norms = True
    except IOError:
      surrogate_load_info += 'Norm fits not found, '
      self.norms = False

    ### empirical time index (ordered by EIM selection) ###
    try:
      self.eim_indices = np.loadtxt(sdir+self._eim_indices_txt,dtype=int)
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
class TextSurrogateWrite(SurrogateBaseIO):
  """Export single-mode, text-based surrogate"""
  __doc__+=surrogate_description

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __init__(self, sdir):
    """open single-mode surrogate, to be located in directory sdir, for writing"""

    if( not(sdir[-1:] is '/') ):
      raise Exception("path name should end in /")
    try:
      os.mkdir(sdir)
      print("Successfully created a surrogate directory...use write_text to export your surrogate!")
    except OSError:
      print("Could not create a surrogate directory. Not ready to export, please try again.")

    ### sdir is defined to be the surrogate's ID ###
    self.SurrogateID = sdir


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def write_text(self, time_info, B, eim_indices, greedy_points, \
                 fit_interval, affine_map, \
                 fitparams_amp, fitparams_phase, fitparams_norm, V, R, \
                 fit_type_phase,\
                 fit_type_amp, fit_type_norm,parameterization,\
                 surrogate_mode_type):
    """ Write surrogate data (text) in standard format."""
		

    # TODO: flag to zip folder and save full time series

    ### pack mass ratio interval (for fits) and time info ###
    self._np_savetxt_safe(self.SurrogateID+self._fit_interval_txt,fit_interval)
    self._np_savetxt_safe(self.SurrogateID+self._time_info_txt,time_info)
    self._np_savetxt_safe(self.SurrogateID+self._greedy_points_txt,\
                         greedy_points,fmt='%2.16f')
    self._np_savetxt_safe(self.SurrogateID+self._eim_indices_txt,\
                         eim_indices,fmt='%i')
    self._np_savetxt_safe(self.SurrogateID+self._B_1_txt,B.real)
    self._np_savetxt_safe(self.SurrogateID+self._B_2_txt,B.imag)
    self._np_savetxt_safe(self.SurrogateID+self._fitparams_phase_txt,\
                         fitparams_phase)
    self._np_savetxt_safe(self.SurrogateID+self._fitparams_amp_txt,\
                         fitparams_amp)
    #self._np_savetxt_safe(self.SurrogateID+self._affine_map_txt,\
    #                     np.array([int(affine_map)]),fmt='%i')
    self._np_savetxt_safe(self.SurrogateID+self._affine_map_txt,\
                         [affine_map],'%s')
    self._np_savetxt_safe(self.SurrogateID+self._V_1_txt,V.real)
    self._np_savetxt_safe(self.SurrogateID+self._V_2_txt,V.imag)
    self._np_savetxt_safe(self.SurrogateID+self._R_1_txt,R.real)
    self._np_savetxt_safe(self.SurrogateID+self._R_2_txt,R.imag)
    self._np_savetxt_safe(self.SurrogateID+self._fitparams_norm_txt,\
                         fitparams_norm)
    self._np_savetxt_safe(self.SurrogateID+self._fit_type_phase_txt,\
                         [fit_type_phase],'%s')
    self._np_savetxt_safe(self.SurrogateID+self._fit_type_amp_txt,\
                         [fit_type_amp],'%s')
    self._np_savetxt_safe(self.SurrogateID+self._fit_type_norm_txt,\
                         [fit_type_norm],'%s')
    self._np_savetxt_safe(self.SurrogateID+self._parameterization_txt,\
                         [parameterization],'%s')
    self._np_savetxt_safe(self.SurrogateID+self._surrogate_mode_type_txt,\
                         [surrogate_mode_type],'%s')

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def _np_savetxt_safe(self,fname, X, fmt='%.18e', delimiter=' ',\
                      newline='\n', header='', footer='', comments='# '):
    """ numpys savetext without overwrites """

    if os.path.isfile(fname):
      raise Exception("file already exists")
    else: 
      np.savetxt(fname,X,fmt=fmt)



