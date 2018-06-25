""" Gravitational Wave Surrogate classes for text and hdf5 files"""

from __future__ import division # for python 2

__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@umassd.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Scott Field, Chad Galley, Vijay Varma"

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

# adding "_" prefix to potentially unfamiliar module names
# so they won't show up in gws' tab completion
import numpy as np
from scipy.interpolate import splrep as _splrep
from scipy.interpolate import splev as _splev
from .gwtools.harmonics import sYlm as _sYlm
from .gwtools import plot_pretty as _plot_pretty
from .gwtools import gwtools as _gwtools # from the package gwtools, import the module gwtools (gwtools.py)....
from .parametric_funcs import function_dict as my_funcs
from .surrogateIO import H5Surrogate as _H5Surrogate
from .surrogateIO import TextSurrogateRead as _TextSurrogateRead
from .surrogateIO import TextSurrogateWrite as _TextSurrogateWrite
from gwsurrogate.new.surrogate import ParamDim, ParamSpace

import warnings
import os

from new import surrogate as new_surrogate
import catalog

try:
  import matplotlib.pyplot as plt
except:
  print("Cannot load matplotlib.")

try:
  import h5py
  h5py_enabled = True
except ImportError:
  h5py_enabled = False


# needed to search for single mode surrogate directories 
def _list_folders(path,prefix):
  '''returns all folders which begin with some prefix'''
  import os as os
  for f in os.listdir(path):
    if f.startswith(prefix):
      yield f

# handy helper to save waveforms 
def write_waveform(t, hp, hc, filename='output',ext='bin'):
  """write waveform to text or numpy binary file"""

  if( ext == 'txt'):
    np.savetxt(filename, [t, hp, hc])
  elif( ext == 'bin'):
    np.save(filename, [t, hp, hc])
  else:
    raise ValueError('not a valid file extension')


##############################################
class ExportSurrogate(_H5Surrogate, _TextSurrogateWrite):
	"""Export single-mode surrogate"""
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path):
		
		# export HDF5 or Text surrogate data depending on input file extension
		ext = path.split('.')[-1]
		if ext == 'hdf5' or ext == 'h5':
			_H5Surrogate.__init__(self, file=path, mode='w')
		else:
			raise ValueError('use TextSurrogateWrite instead')


##############################################
class EvaluateSingleModeSurrogate(_H5Surrogate, _TextSurrogateRead):
  """Evaluate single-mode surrogate in terms of the waveforms' amplitude and phase"""


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __init__(self, path, deg=3, subdir='', closeQ=True):

    # Load HDF5 or Text surrogate data depending on input file extension
    if type(path) == h5py._hl.files.File:
      ext = 'h5'
    else:
      ext = path.split('.')[-1]
    if ext == 'hdf5' or ext == 'h5':
      _H5Surrogate.__init__(self, file=path, mode='r', subdir=subdir, closeQ=closeQ)
    else:
      _TextSurrogateRead.__init__(self, path)
    
    # Interpolate columns of the empirical interpolant operator, B, using cubic spline
    if self.surrogate_mode_type  == 'waveform_basis':
      self.reB_spline_params = [_splrep(self.times, self.B[:,jj].real, k=deg) for jj in range(self.B.shape[1])]
      self.imB_spline_params = [_splrep(self.times, self.B[:,jj].imag, k=deg) for jj in range(self.B.shape[1])]
    elif self.surrogate_mode_type  == 'amp_phase_basis':
      self.B1_spline_params = [_splrep(self.times, self.B_1[:,jj], k=deg) for jj in range(self.B_1.shape[1])]
      self.B2_spline_params = [_splrep(self.times, self.B_2[:,jj], k=deg) for jj in range(self.B_2.shape[1])]
    else:
      raise ValueError('invalid surrogate type')

    pass

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __call__(self, q, M=None, dist=None, phi_ref=None,\
                     f_low=None, times=None, units='dimensionless',\
                     singlemode_call=True):
    """Return single mode surrogate evaluation for...

       Input
       =====
       q               --- binary parameter values EXCLUDING total mass M.
                           In 1D, mass ratio (dimensionless) must be supplied.
                           In nD, the surrogate's internal parameterization is assumed.
       M               --- total mass (solar masses) 
       dist            --- distance to binary system (megaparsecs)
       phi_ref         --- mode's phase phi(t), as h=A*exp(i*phi) at peak amplitude
       f_low           --- instantaneous initial frequency, will check if flow_surrogate < f_low
       times           --- array of times at which surrogate is to be evaluated
       units           --- units (mks or dimensionless) of input array of time samples
       singlemode_call --- Never set this by hand, will be False if this routine is called by the multimode evaluator


       More information
       ================
       This routine evaluates gravitational wave complex polarization modes h_{ell m}
       defined on a sphere whose origin is the binary's center of mass. 

       Dimensionless surrogates rh/M are evaluated by calling _h_sur. 
       Physical surrogates are generated by applying additional operations/scalings.

       If M and dist are provided, a physical surrogate will be returned in mks units.

       An array of times can be passed along with its units. """

    # surrogate evaluations assumed dimensionless, physical modes are found from scalings 
    if self.surrogate_units != 'dimensionless':
      raise ValueError('surrogate units is not supported')

    if (units != 'dimensionless') and (units != 'mks'):
      raise ValueError('units is not supported')

    ### if (M,distance) provided, a physical mode in mks units is returned ###
    if( M is not None and dist is not None):
      amp0    = ((M * _gwtools.MSUN_SI ) / (dist * _gwtools.PC_SI )) * ( _gwtools.G / np.power(_gwtools.c,2.0) )
      t_scale = _gwtools.Msuninsec * M
    else:
      amp0    = 1.0
      t_scale = 1.0

    ### evaluation times t: input times or times at which surrogate was built ###
    if (times is not None):
      t = times
    else:
      t = self.time()

    ### if input times are dimensionless, convert to MKS if a physical surrogate is requested ###
    if units == 'dimensionless':
      t = t_scale * t

    # because times is passed to _h_sur, it must be dimensionless form t/M
    if times is not None and units == 'mks':
      times = times / t_scale

    # convert from input to internal surrogate parameter values, and check within training region #
    x = self.get_surr_params_safe(q)

    ### Evaluate dimensionless single mode surrogates ###
    hp, hc = self._h_sur(x, times=times)


    ### adjust mode's phase by an overall constant ###
    if (phi_ref is not None):
      h  = self.adjust_merger_phase(hp + 1.0j*hc,phi_ref)
      hp = h.real
      hc = h.imag

    ### Restore amplitude scaling ###
    hp     = amp0 * hp
    hc     = amp0 * hc

    ### check that surrogate's starting frequency is below f_low, otherwise throw a warning ###
    if f_low is not None:
      self.find_instant_freq(hp, hc, t, f_low)

    return t, hp, hc


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def find_instant_freq(self, hp, hc, t, f_low = None):
    """instantaneous frequency at t_start for 

          h = A(t) exp(2 * pi * i * f(t) * t), 

       where \partial_t A ~ \partial_t f ~ 0. If f_low passed will check its been achieved."""

    f_instant = _gwtools.find_instant_freq(hp, hc, t)

    # TODO: this is a hack to account for inconsistent conventions!
    f_instant = np.abs(f_instant)

    if f_low is None:
      return f_instant
    else:
      if f_instant > f_low:
        raise Warning("starting frequency is "+str(f_instant))
      else:
        pass


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def amp_phase(self,h):
    """Get amplitude and phase of waveform, h = A*exp(i*phi)"""
    return _gwtools.amp_phase(h)


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def phi_merger(self,h):
    """Phase of mode at amplitude's discrete peak. h = A*exp(i*phi)."""

    amp, phase = self.amp_phase(h)
    argmax_amp = np.argmax(amp)

    return phase[argmax_amp]


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def adjust_merger_phase(self,h,phiref):
    """Modify GW mode's phase such that at time of amplitude peak, t_peak, we have phase(t_peak) = phiref"""

    phimerger = self.phi_merger(h)
    phiadj    = phiref - phimerger

    return _gwtools.modify_phase(h,phiadj)


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def timer(self,M_eval=None,dist_eval=None,phi_ref=None,f_low=None,times=None):
    """average time to evaluate surrogate waveforms. """

    qmin, qmax = self.fit_interval
    ran = np.random.uniform(qmin, qmax, 1000)

    import time
    tic = time.time()
    if M_eval is None:
      for i in ran:
        hp, hc = self._h_sur(i)
    else:
      for i in ran:
        t, hp, hc = self.__call__(i,M_eval,dist_eval,phi_ref,f_low,times)

    toc = time.time()
    print('Timing results (results quoted in seconds)...')
    print('Total time to generate 1000 waveforms = ',toc-tic)
    print('Average time to generate a single waveform = ', (toc-tic)/1000.0)
    pass

	
  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def time(self, units=None,M=None,dt=None):
    """Return the set of time samples at which the surrogate is defined.
       If dt is supplied, return the requested uniform grid.

      INPUT (all optional)
      =====
      units --- None:        time in geometric units, G=c=1
                'mks'        time in seconds
                'solarmass': time in solar masses
      M     --- Mass (in units of solar masses).
      dt    --- delta T

      OUTPUT
      ======
      1) units = M = None:   Return time samples at which the surrogate as built for.
      2) units != None, M=:  Times after we convert from surrogate's self.t_units to units.
                             If units = 'mks' and self.t_units='TOverMtot' then M must
                             be supplied to carry out conversion.
      3) dt != None:         Return time grid as np.arange(t[0],t[-1],dt)"""


    if units is None:
      t = self.times
    elif (units == 'mks') and (self.t_units == 'TOverMtot'):
      assert(M!=None)
      t = (_gwtools.Msuninsec*M) * self.times
    else:
      raise ValueError('Cannot compute times')

    if dt is None:
      return t
    else:
      return np.arange(t[0],t[-1],dt)

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def basis(self, i, flavor='waveform'):
    """compute the ith cardinal, orthogonal, or waveform basis."""

    if self.surrogate_mode_type  == 'waveform_basis':

      if flavor == 'cardinal':
        basis = self.B[:,i]
      elif flavor == 'orthogonal':
        basis = np.dot(self.B,self.V)[:,i]
      elif flavor == 'waveform':
        E = np.dot(self.B,self.V)
        basis = np.dot(E,self.R)[:,i]
      else:
        raise ValueError("Not a valid basis type")

    elif self.surrogate_mode_type  == 'amp_phase_basis':
        raise ValueError("Not coded yet")

    return basis

  # TODO: basis resampling should be unified.
  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # TODO: ext should be passed from __call__
  def resample_B(self, times, ext=1):
    """resample the empirical interpolant operator, B, at the input time samples"""

    evaluations = np.array([_splev(times, self.reB_spline_params[jj],ext=ext)  \
             + 1j*_splev(times, self.imB_spline_params[jj],ext=ext) for jj in range(self.B.shape[1])]).T

    # allow for extrapolation if very close to surrogate's temporal interval
    t0 = self.times[0]
    if (np.abs(times[0] - t0) < t0 * 1.e-12) or (t0==0 and np.abs(times[0] - t0) <1.e-12):
      evaluations[0] = np.array([_splev(times[0], self.reB_spline_params[jj],)  \
             + 1j*_splev(times[0], self.imB_spline_params[jj]) for jj in range(self.B.shape[1])]).T
    
    return evaluations

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # TODO: ext should be passed from __call__
  def resample_B_1(self, times, ext=1):
    """resample the B_1 basis at the input time samples"""

    evaluations = np.array([_splev(times, self.B1_spline_params[jj],ext=1) for jj in range(self.B_1.shape[1])]).T

    # allow for extrapolation if very close to surrogate's temporal interval
    if np.abs(times[0] - self.times[0])/self.times[0] < 1.e-12:
      evaluations[0] = np.array([_splev(times[0], self.B1_spline_params[jj],ext=1) for jj in range(self.B_1.shape[1])]).T
    
    return evaluations

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # TODO: ext should be passed from __call__
  def resample_B_2(self, times, ext=1):
    """resample the B_2 basis at the input samples"""

    evaluations = np.array([_splev(times, self.B2_spline_params[jj],ext=1) for jj in range(self.B_2.shape[1])]).T

    # allow for extrapolation if very close to surrogate's temporal interval
    if np.abs(times[0] - self.times[0])/self.times[0] < 1.e-12:
      evaluations[0] = np.array([_splev(times[0], self.B2_spline_params[jj],ext=1) for jj in range(self.B_2.shape[1])]).T
    
    return evaluations


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def plot_rb(self, i, showQ=True):
    """plot the ith reduced basis waveform"""

    # Compute surrogate approximation of RB waveform
    basis = self.basis(i)
    fig   = _plot_pretty(self.times,[basis.real,basis.imag])

    if showQ:
      plt.show()
    
    # Return figure method to allow for saving plot with fig.savefig
    return fig


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def plot_sur(self, q_eval, timeM=False, htype='hphc', flavor='linear', color='k', linestyle=['-', '--'], \
                label=['$h_+(t)$', '$h_-(t)$'], legendQ=False, showQ=True):
    """plot surrogate evaluated at mass ratio q_eval"""

    t, hp, hc = self.__call__(q_eval)
    h = hp + 1j*hc
    
    y = {
      'hphc': [hp, hc],
      'hp': hp,
      'hc': hc,
      'AmpPhase': [np.abs(h), _gwtools.phase(h)],
      'Amp': np.abs(h),
      'Phase': _gwtools.phase(h),
      }
    
    if self.t_units == 'TOverMtot':
      xlab = 'Time, $t/M$'
    else:
      xlab = 'Time, $t$ (sec)'

    # Plot surrogate waveform
    fig = _plot_pretty(t, y[htype], flavor=flavor, color=color, linestyle=linestyle, \
                label=label, legendQ=legendQ, showQ=False)
    plt.xlabel(xlab)
    plt.ylabel('Surrogate waveform')
    
    if showQ:
      plt.show()
        
    # Return figure method to allow for saving plot with fig.savefig
    return fig
  
  
  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def plot_eim_data(self, inode=None, htype='Amp', nuQ=False, fignum=1, showQ=True):
    """Plot empirical interpolation data used for performing fits in parameter"""
    
    fig = plt.figure(fignum)
    ax1 = fig.add_subplot(111)
    
    y = {
      'Amp': self.eim_amp,
      'Phase': self.eim_phase,
      }
    
    if nuQ:
      nu = _gwtools.q_to_nu(self.greedy_points)
      
      if inode is None:
        [plt.plot(nu, ee, 'ko') for ee in y[htype]]
      else:
        plt.plot(nu, y[htype][inode], 'ko')
      
      plt.xlabel('Symmetric mass ratio, $\\nu$')
    
    else:
      
      if inode is None:
        [plt.plot(self.greedy_points, ee, 'ko') for ee in y[htype]]
      else:
        plt.plot(self.greedy_points, y[htype][inode], 'ko')
      
      plt.xlabel('Mass ratio, $q$')
    
    if showQ:
      plt.show()
    
    return fig
  
  
  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def plot_eim_fits(self, inode=None, htype='Amp', nuQ=False, fignum=1, num=200, showQ=True):
    """Plot empirical interpolation data and fits"""
    
    fig = plt.figure(fignum)
    ax1 = fig.add_subplot(111)
    
    fitfn = {
      'Amp': self.amp_fit_func,
      'Phase': self.phase_fit_func,
      }
    
    coeffs = {
      'Amp': self.fitparams_amp,
      'Phase': self.fitparams_phase,
      }
    
    # Plot EIM data points
    self.plot_eim_data(inode=inode, htype=htype, nuQ=nuQ, fignum=fignum, showQ=False)
    
    qs = np.linspace(self.fit_min, self.fit_max, num)
    nus = _gwtools.q_to_nu(qs)
    
    # Plot fits to EIM data points
    if nuQ:
      if inode is None:
        [plt.plot(nus, fitfn[htype](cc, qs), 'k-') for cc in coeffs[htype]]
      else:
        plt.plot(nus, fitfn[htype](coeffs[htype][inode], qs), 'k-')  
    
    else:
      if inode is None:
        [plt.plot(qs, fitfn[htype](cc, qs), 'k-') for cc in coeffs[htype]]
      else:
        plt.plot(qs, fitfn[htype](coeffs[htype][inode], qs), 'k-')
      
    if showQ:
      plt.show()
    
    return fig
  

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # TODO: strong_checking should be kwargs
  # TODO: only check once in multimode surrogates
  def check_training_interval(self, x, strong_checking=True):
    """Check if parameter value x is within the training interval."""

    x_min, x_max = self.fit_interval

    if(np.any(x < x_min) or np.any(x > x_max)):
      if strong_checking:
        raise ValueError('Surrogate not trained at requested parameter value')
      else:
        print("Warning: Surrogate not trained at requested parameter value")
        Warning("Surrogate not trained at requested parameter value")


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def get_surr_params_safe(self,x):
    """ Compute the surrogate's *internal* parameter values from the input ones, x,
        passed to __call__ with safe bounds checking.

        The function get_surr_params used in the conversion is set in SurrogateIO
        as specified by the surrogate's data value corresponding to the key PARAMETERIZATION.
        Therefore, SurrogateIO must be aware of what x is expected to be. 

          Example: The user may pass mass ratio q=x to __call__, but the
                   symmetric mass ratio x_internal = q / (1+q)^2 might parameterize the surrogate

        After the parameter change of coordinates is done, check its within the surrogate's
        training region. A training region is assumed to be an n-dim rectangle.

        x is assumed to NOT have total mass M as a parameter. ``Bare" surrogates are always dimensionless."""

    x_internal = self.get_surr_params(x)

    # TODO: this will (redundantly) check for each mode. Multimode surrogate should directly check it
    self.check_training_interval(x_internal)

    return x_internal

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def norm_eval(self, x):
    """Evaluate norm fit at parameter value x.

       Wrapper for norm evaluations called from outside of the class"""

    self.check_training_interval(x, strong_checking=True)
    x_0 = self._affine_mapper(x) # _norm_eval won't do its own affine mapping
    return self._norm_eval(x_0)

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def eim_coeffs(self, x, surrogate_mode_type):
    """Evaluate EIM coefficients at parameter value x.

       Wrapper for safe calls from outside of the class"""

    self.check_training_interval(x, strong_checking=True)
    return self._eim_coeffs(x, surrogate_mode_type)

  #### below here are "private" member functions ###
  # These routine's evaluate a "bare" surrogate, and should only be called
  # by the __call__ method 
  #
  # These routine's use x as the parameter, which could be mass ratio,
  # symmetric mass ratio, or something else. Parameterization info should
  # be supplied by surrogate's parameterization tag.

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def _affine_mapper(self, x):
    """map parameter value x to the standard interval [-1,1] if necessary."""

    x_min, x_max = self.fit_interval

    if self.affine_map == 'minus1_to_1':
      x_0 = 2.*(x - x_min)/(x_max - x_min) - 1.;
    elif self.affine_map == 'zero_to_1':
      x_0 = (x - x_min)/(x_max - x_min);
    elif self.affine_map == 'none':
      x_0 = x
    else:
      raise ValueError('unknown affine map')
    return x_0


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def _norm_eval(self, x_0):
    """Evaluate norm fit at x_0, where x_0 is the mapped parameter value.

       WARNING: this function should NEVER be called from outside the class."""

    if not self.norms:
      return 1.
    else:
      return np.array([ self.norm_fit_func(self.fitparams_norm, x_0) ])


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def _amp_eval(self, x_0):
    """Evaluate set of amplitude fits at x_0, where x_0 is the mapped parameter value.

       WARNING: this function should NEVER be called from outside the class."""

    if self.fit_type_amp == 'fast_spline_real':
      return self.amp_fit_func(self.fitparams_amp, x_0)
    else:
      return np.array([ self.amp_fit_func(self.fitparams_amp[jj,:], x_0) for jj in range(self.fitparams_amp.shape[0]) ])


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def _phase_eval(self, x_0):
    """Evaluate set of phase fit at x_0, where x_0 is the mapped parameter value.

       WARNING: this function should NEVER be called from outside the class."""

    if self.fit_type_phase == 'fast_spline_imag':
      return self.phase_fit_func(self.fitparams_phase, x_0)
    else:
      return np.array([ self.phase_fit_func(self.fitparams_phase[jj,:], x_0) for jj in range(self.fitparams_phase.shape[0]) ])


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def _eim_coeffs(self, x, surrogate_mode_type):
    """Evaluate EIM coefficients at parameter value x. x could be mass ratio, symmetric
       mass ratio or something else -- it depends on the surrogate's parameterization. 

       see __call__ for the parameterization and _h_sur for how these 
       coefficients are used. 

       If called from outside the class, check_training_interval should be used
       to determine whether x is in the training interval.

       WARNING: this function should NEVER be called from outside the class."""


    ### x to the standard interval on which the fits were performed ###
    x_0 = self._affine_mapper(x)

    ### Evaluate amp/phase/norm fits ###
    amp_eval   = self._amp_eval(x_0)
    phase_eval = self._phase_eval(x_0)
    nrm_eval   = self._norm_eval(x_0)

    if self.surrogate_mode_type  == 'waveform_basis':
      if self.fit_type_amp == 'fast_spline_real':
        h_EIM = nrm_eval * (amp_eval + 1j*phase_eval)
      else:
        h_EIM = nrm_eval*amp_eval*np.exp(1j*phase_eval) # dim_RB-vector fit evaluation of h
      return h_EIM
    elif self.surrogate_mode_type  == 'amp_phase_basis':
      if self.fit_type_amp == 'fast_spline_real':
        raise ValueError("invalid combination")
      return amp_eval, phase_eval, nrm_eval
    else: 
      raise ValueError('invalid surrogate type')

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def _h_sur(self, x, times=None):
    """Evaluate surrogate at parameter value x. x could be mass ratio, symmetric
       mass ratio or something else -- it depends on the surrogate's parameterization. 

       Returns dimensionless rh/M waveforms in units of t/M.

       This should ONLY be called by the __call__ method which accounts for 
       different parameterization choices. """


    if self.surrogate_mode_type  == 'waveform_basis':

      h_EIM = self._eim_coeffs(x, 'waveform_basis')

      if times is None:
        surrogate = np.dot(self.B, h_EIM)
      else:
        surrogate = np.dot(self.resample_B(times), h_EIM)

      #surrogate = nrm_eval * surrogate

    elif self.surrogate_mode_type  == 'amp_phase_basis':

      amp_eval, phase_eval, nrm_eval = self._eim_coeffs(x, 'amp_phase_basis')

      if times is None:
        sur_A = np.dot(self.B_1, amp_eval)
        sur_P = np.dot(self.B_2, phase_eval)
      else:
        #sur_A = np.dot(np.array([_splev(times, self.B1_spline_params[jj],ext=1) for jj in range(self.B_1.shape[1])]).T, amp_eval)
        #sur_P = np.dot(np.array([_splev(times, self.B2_spline_params[jj],ext=1) for jj in range(self.B_2.shape[1])]).T, phase_eval)
        sur_A = np.dot(self.resample_B_1(times), amp_eval)
        sur_P = np.dot(self.resample_B_2(times), phase_eval)

      surrogate = nrm_eval * sur_A * np.exp(1j*sur_P)


    else:
      raise ValueError('invalid surrogate type')


    hp = surrogate.real
    #hp = hp.reshape([self.time_samples,])
    hc = surrogate.imag
    #hc = hc.reshape([self.time_samples,])

    return hp, hc


def CreateManyEvaluateSingleModeSurrogates(path, deg, ell_m, excluded, enforce_orbital_plane_symmetry):
  """For each surrogate mode an EvaluateSingleModeSurrogate class
     is created.

     INPUT
     =====
     path: the path to the surrogate
     deg: the degree of the splines representing the basis (default 3, cubic)
     ell_m: A list of (ell, m) modes to load, for example [(2,2),(3,3)].
            None (default) loads all modes.
     excluded: A list of (ell, m) modes to skip loading.
        The default ('DEFAULT') excludes any modes with an 'EXCLUDED' dataset.
        Use [] or None to load these modes as well.
     enforce_orbital_plane_symmetry: If set to True an exception is raised if the 
        surrogate data contains negative modes. This can be used to gaurd against
        mixing spin-aligned and precessing surrogates...which have different
        evaluation patterns for m<0.

     Returns single_mode_dict. Keys are (ell, m) mode and value is an
     instance of EvaluateSingleModeSurrogate."""


  if excluded is None:
    excluded = []

  ### fill up dictionary with single mode surrogate class ###
  single_mode_dict = dict()

  # Load HDF5 or Text surrogate data depending on input file extension
  if type(path) == h5py._hl.files.File:
    ext = 'h5'
    filemode = path.mode
  else:
    ext = path.split('.')[-1]
    filemode = 'r'

  # path, excluded 
  if ext == 'hdf5' or ext == 'h5':
      
    if filemode not in ['r+', 'w']:
      fp = h5py.File(path, filemode)

      ### compile list of excluded modes ###
      if type(excluded) == list:
        exc_modes = excluded
      elif excluded == 'DEFAULT':
        exc_modes = []
      else:
        raise ValueError('Invalid excluded option: %s'%excluded)
      for kk, vv in fp.items():  # inefficient on Py2
        if 'EXCLUDED' in vv:
          splitkk = kk.split('_')
          if splitkk[0][0] == 'l' and splitkk[1][0] == 'm':
            ell = int(splitkk[0][1])
            emm = int(splitkk[1][1:])
            if excluded == 'DEFAULT':
              exc_modes.append((ell,emm))
            elif not (ell, emm) in exc_modes:
              print("Warning: Including mode (%d,%d) which is excluded by default"%(ell, emm))
       ### compile list of available modes ###
      if ell_m is None:
        mode_keys = []
        for kk in fp.keys(): # Py2 list, Py3 iterator
          splitkk = kk.split('_')
          if splitkk[0][0] == 'l' and splitkk[1][0] == 'm':
            ell = int(splitkk[0][1])
            emm = int(splitkk[1][1:])
            if not (ell, emm) in exc_modes:
              mode_keys.append((ell,emm))
      else:
        mode_keys = []
        for i, mode in enumerate(ell_m):
          if mode in exc_modes:
            print("WARNING: Mode (%d,%d) is both included and excluded! Excluding it."%mode) 
          else:
            mode_keys.append(mode)

      # If we are using orbital symmetry, make sure we aren't loading any negative m modes
      if enforce_orbital_plane_symmetry:
        for ell, emm in mode_keys:
          if emm < 0:
            raise Exception("When using enforce_orbital_plane_symmetry, do not load negative m modes!")

       ### load the single mode surrogates ###
      for mode_key in mode_keys:
        assert(mode_keys.count(mode_key)==1)
        mode_key_str = 'l'+str(mode_key[0])+'_m'+str(mode_key[1])
        print("loading surrogate mode... " + mode_key_str)
        single_mode_dict[mode_key] = \
          EvaluateSingleModeSurrogate(fp,subdir=mode_key_str+'/',closeQ=False)
      fp.close()

  else:
    ### compile list of available modes ###
    # assumes (i) single mode folder format l#_m#_ 
    #         (ii) ell<=9, m>=0
    import os
    for single_mode in _list_folders(path,'l'):
      ell = int(single_mode[1])
      emm = int(single_mode[4])
      mode_key = (ell,emm)
      if (ell_m is None) or (mode_key in ell_m):
        if ((type(excluded) == list and not mode_key in excluded) or
            (excluded == 'DEFAULT' and not
             os.path.isfile(path+single_mode+'/EXCLUDED.txt'))):
          assert(mode_key not in single_mode_dict)
          if os.path.isfile(path+single_mode+'/EXCLUDED.txt'):
            print("Warning: Including mode (%d,%d) which is excluded by default"%(ell, emm))
          if enforce_orbital_plane_symmetry and emm < 0:
            raise Exception("When using enforce_orbital_plane_symmetry, do not load negative m modes!")

          print("loading surrogate mode... "+single_mode[0:5])
          single_mode_dict[mode_key] = \
            EvaluateSingleModeSurrogate(path+single_mode+'/')
    ### check all requested modes have been loaded ###
    if ell_m is not None:
      for tmp in ell_m:
        try:
          single_mode_dict[tmp]
        except KeyError:
          print('Could not find mode '+str(tmp))
 
  return single_mode_dict


##############################################
class EvaluateSurrogate():
  """Evaluate multi-mode surrogates"""

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __init__(self, path, deg=3, ell_m=None, excluded='DEFAULT', use_orbital_plane_symmetry=True):
    """Loads a surrogate.

    path: the path to the surrogate
    deg: the degree of the splines representing the basis (default 3, cubic). 
        Unless there is good reason to use deg !=3 one should not change this.
        Some surrogates (e.g. 4d2s) are validated with this in mind.
    ell_m: A list of (ell, m) modes to load, for example [(2,2),(3,3)].
        None (default) loads all modes.
    excluded: A list of (ell, m) modes to skip loading.
        The default ('DEFAULT') excludes any modes with an 'EXCLUDED' dataset.
        Use [] or None to load these modes as well.
    use_orbital_plane_symmetry: If set to true (i) CreateManyEvaluateSingleModeSurrogates
        will explictly check that m<0 do not exist in the data file and (ii) m<0 modes
        are inferred from m>0 modes. If set to false no symmetry is assumed -- typical
        of precessing models. When False, fake_neg_modes must be false."""

    self.single_mode_dict = \
      CreateManyEvaluateSingleModeSurrogates(path, deg, ell_m, excluded, use_orbital_plane_symmetry)

    self.use_orbital_plane_symmetry = use_orbital_plane_symmetry

    ### Load/deduce multi-mode surrogate properties ###
    #if filemode not in ['r+', 'w']:      
    if len(self.single_mode_dict) == 0:
      raise IOError('Modes not found. Mode subdirectories begins with l#_m#_')

    
    first_mode_surr = self.single_mode_dict[list(self.single_mode_dict.keys())[0]]
                                            
    ### Check single mode temporal grids are collocated -- define common grid ###
    grid_shape = first_mode_surr.time().shape
    for key in list(self.single_mode_dict.keys()):
      tmp_shape = self.single_mode_dict[key].time().shape
      if(grid_shape != tmp_shape):
        raise ValueError('inconsistent single mode temporal grids')  
        
    # common time grid for all modes
    self.time_grid = first_mode_surr.time

    ### Check single mode surrogates have the same parameterization ###
    # TODO: if modes use different parameterization -- better to let modes handle this?
    training_parameter_range = first_mode_surr.fit_interval
    parameterization = first_mode_surr.get_surr_params
    for key in list(self.single_mode_dict.keys()):
      tmp_range = self.single_mode_dict[key].fit_interval
      tmp_parameterization = self.single_mode_dict[key].get_surr_params
      if(np.max(np.abs(tmp_range - training_parameter_range)) != 0):
        raise ValueError('inconsistent single mode parameter grids')  
      if(tmp_parameterization != parameterization):
        raise ValueError('inconsistent single mode parameterizations') 
    # common parameter interval and parameterization for all modes 
    # use newer parameter space class for common interface
    pd = ParamDim(name='unknown parmater', 
                  min_val=training_parameter_range[0],
                  max_val=training_parameter_range[1])
    self.param_space = ParamSpace(name='unknown', params=[pd])
    self.parameterization = parameterization
    
    print("Surrogate interval",training_parameter_range)
    print("Surrogate time grid",self.time_grid())
    print("Surrogate parameterization"+self.parameterization.__doc__)

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def __call__(self, q, M=None, dist=None, theta=None,phi=None,
                     z_rot=None, f_low=None, times=None,
                     units='dimensionless',
                     ell=None, m=None, mode_sum=True,fake_neg_modes=True):
    """Return surrogate evaluation for...

      INPUT
      =====
      q              --- binary parameter values EXCLUDING total mass M.
                         In 1D, mass ratio (dimensionless) must be supplied.
                         In nD, the surrogate's internal parameterization is assumed.
      M              --- total mass (solar masses) 
      dist           --- distance to binary system (megaparsecs)
      theta/phi      --- evaluate hp and hc modes at this location on sphere
      z_rot          --- physical rotation about angular momentum (z-)axis (radians)
      flow           --- instantaneous initial frequency, will check if flow_surrogate < flow mode-by-mode
      times          --- array of times at which surrogate is to be evaluated
      units          --- units ('mks' or 'dimensionless') of input array of time samples
      ell            --- list or array of N ell modes to evaluate for (if none, all modes are returned)
      m              --- for each ell, supply a matching m value 
      mode_sum       --- if true, all modes are summed, if false all modes are returned in an array
      fake_neg_modes --- if true, include m<0 modes deduced from m>0 mode. all m in [ell,m] input should be non-negative

      NOTE: if only requesting one mode, this should be ell=[2],m=[2]

       Note about Angles
       =================
       For circular orbits, the binary's orbital angular momentum is taken to
       be the z-axis. Theta and phi is location on the sphere relative to this 
       coordinate system. """


    if (not self.use_orbital_plane_symmetry) and fake_neg_modes:
      raise ValueError("if use_orbital_plane_symmetry is not assumed, it is not possible to fake m<0 modes")

    ### deduce single mode dictionary keys from ell, m and fake_neg_modes input ###
    modes_to_evaluate = self.generate_mode_eval_list(ell,m,fake_neg_modes)

    if mode_sum and (theta is None and phi is None) and len(modes_to_evaluate)!=1:
      raise ValueError('Trying to sum modes without theta and phi is a strange idea')

    ### if mode_sum false, return modes in a sensible way ###
    if not mode_sum:
      modes_to_evaluate = self.sort_mode_list(modes_to_evaluate)

    # Modes actually modeled by the surrogate. We will fake negative m
    # modes later if needed. 
    modeled_modes = self.all_model_modes(False)

    ### allocate arrays for multimode polarizations ###
    if mode_sum:
      hp_full, hc_full = self._allocate_output_array(times,1,mode_sum)
    else:
      hp_full, hc_full = self._allocate_output_array(times,len(modes_to_evaluate),mode_sum)

    ### loop over all evaluation modes ###
    # TODO: internal workings are simplified if h used instead of (hc,hp)
    ii = 0
    for ell,m in modes_to_evaluate:

      ### if the mode is modelled, evaluate it. Otherwise its zero ###
      is_modeled = (ell,m) in modeled_modes
      neg_modeled = (ell,-m) in modeled_modes
      if is_modeled or (neg_modeled and fake_neg_modes):

        if is_modeled:
          t_mode, hp_mode, hc_mode = self.evaluate_single_mode(q,M,dist,f_low,times,units,ell,m)
        else: # then we must have neg_modeled=True and fake_neg_modes=True
          t_mode, hp_mode, hc_mode = self.evaluate_single_mode_by_symmetry(q,M,dist,f_low,times,units,ell,m)

        if z_rot is not None:
          h_tmp   = _gwtools.modify_phase(hp_mode+1.0j*hc_mode,z_rot*m)
          hp_mode = h_tmp.real
          hc_mode = h_tmp.imag

        # TODO: should be faster. integrate this later on
        #if fake_neg_modes and m != 0:
        #  hp_mode_mm, hc_mode_mm = self._generate_minus_m_mode(hp_mode,hc_mode,ell,m)
        #  hp_mode_mm, hc_mode_mm = self.evaluate_on_sphere(ell,-m,theta,phi,hp_mode_mm,hc_mode_mm)

        hp_mode, hc_mode = self.evaluate_on_sphere(ell,m,theta,phi,hp_mode,hc_mode)

        if mode_sum:
          hp_full = hp_full + hp_mode
          hc_full = hc_full + hc_mode
          #if fake_neg_modes and m != 0:
          #  hp_full = hp_full + hp_mode_mm
          #  hc_full = hc_full + hc_mode_mm
        else:
          if len(modes_to_evaluate)==1:
            hp_full[:] = hp_mode[:]
            hc_full[:] = hc_mode[:]
          else: 
            hp_full[:,ii] = hp_mode[:]
            hc_full[:,ii] = hc_mode[:]
      else:
        warning_str = "Your mode (ell,m) = ("+str(ell)+","+str(m)+") is not available!"
        raise Warning(warning_str)

      
      ii+=1

    if mode_sum:
      return t_mode, hp_full, hc_full #assumes all mode's have same temporal grid
    else: # helpful to have (l,m) list for understanding mode evaluations
      return modes_to_evaluate, t_mode, hp_full, hc_full


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def evaluate_on_sphere(self,ell,m,theta,phi,hp_mode,hc_mode):
    """evaluate on the sphere"""

    if theta is not None: 
      #if phi is None: phi = 0.0
      if phi is None: raise ValueError('phi must have a value')
      sYlm_value =  _sYlm(-2,ll=ell,mm=m,theta=theta,phi=phi)
      h = sYlm_value*(hp_mode + 1.0j*hc_mode)
      hp_mode = h.real
      hc_mode = h.imag

    return hp_mode, hc_mode

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def evaluate_single_mode(self,q, M, dist, f_low, times, units,ell,m):
    """ light wrapper around single mode evaluator"""

    t_mode, hp_mode, hc_mode = self.single_mode_dict[(ell,m)](q, M, dist, None, f_low, times,
                                                              units,singlemode_call=False)

    return t_mode, hp_mode, hc_mode


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def evaluate_single_mode_by_symmetry(self,q, M, dist, f_low, times, units,ell,m):
    """ evaluate m<0 mode from m>0 mode and relationship between these"""

    if m<0:
      t_mode, hp_mode, hc_mode = self.evaluate_single_mode(q, M, dist, f_low, times, units,ell,-m)
      hp_mode, hc_mode         = self._generate_minus_m_mode(hp_mode,hc_mode,ell,-m)
    else:
      raise ValueError('m must be negative.')

    return t_mode, hp_mode, hc_mode


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def generate_mode_eval_list(self,ell=None,m=None,minus_m=False):
    """generate list of (ell,m) modes to evaluate for.

      1) ell=m=None: use all available model modes
      2) ell=NUM, m=None: all modes up to ell_max = NUM. unmodelled modes set to zero
      3) list of [ell], [m] pairs: only use modes (ell,m). unmodelled modes set to zero 
         ex: ell=[3,2] and m=[2,2] generates a (3,2) and (2,2) mode.

      These three options produce a list of (ell,m) modes.

      Set minus_m=True to generate m<0 modes from m>0 modes."""

    ### generate list of nonnegative m modes to evaluate for ###
    if ell is None and m is None:
      modes_to_eval = self.all_model_modes()
    elif m is None:
      LMax = ell
      modes_to_eval = []
      for L in range(2,LMax+1):
        for emm in range(0,L+1):
          modes_to_eval.append((L,emm))
    else: # neither pythonic nor fast
      #modes_to_eval = [(x, y) for x in ell for y in m]
      modes_to_eval = []
      for ii in range(len(ell)):
        modes_to_eval.append((ell[ii],m[ii]))

    ### if m<0 requested, build these from m>=0 list ###
    if minus_m:
      modes_to_eval = self._extend_mode_list_minus_m(modes_to_eval)

    return modes_to_eval

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def sort_mode_list(self,mode_list):
    """sort modes as (2,-2), (2,-1), ..., (2,2), (3,-3),(3,-2)..."""

    from operator import itemgetter

    mode_list = sorted(mode_list, key=itemgetter(0,1))
    return mode_list


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def all_model_modes(self,minus_m=False):
    """ from single mode keys deduce all available model modes.
        If minus_m=True, include (ell,-m) whenever (ell,m) is available ."""

    model_modes = [(ell,m) for ell,m in self.single_mode_dict.keys()] # Py2 list, Py3 iterator

    if minus_m:
      model_modes = self._extend_mode_list_minus_m(model_modes)

    return model_modes


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def single_mode(self,mode):
    """ Returns a single-mode object for mode=(ell,m).
        This object stores information for the (ell,m)-mode surrogate"""
    return self.single_mode_dict[mode] 


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def match_surrogate(self, t_ref,h_ref,q, M=None, dist=None, theta=None,\
                     t_ref_units='dimensionless',ell=None,m=None,fake_neg_modes=True,\
                     t_low_adj=.0125,t_up_adj=.0125,speed='slow'):
    """ match discrete complex polarization (t_ref,h_ref) to surrogate waveform for 
        given input values. Inputs have same meaning as those passed to __call__

        Minimization (i.e. match) over time shifts and z-axis rotations"""

    # TODO: routine only works for hp,hc evaluated on the sphere. should extend to modes

    if (M is None and t_ref_units=='mks') or (M is not None and t_ref_units=='dimensionless'):
      raise ValueError('surrogate evaluations and reference temporal grid are inconsistent')


    if speed == 'slow': # repeated calls to surrogate evaluation routines

      ### setup minimization problem -- deduce common time grid and approximate minimization solution from discrete waveform ###
      time_sur,hp,hc = self.__call__(q=q,M=M,dist=dist,theta=theta,phi=0.0,\
                                  times=self.time_all_modes(), units='dimensionless',ell=ell,m=m,fake_neg_modes=fake_neg_modes)
      h_sur =  hp + 1.0j*hc

      # TODO: this deltaPhi is overall phase shift -- NOT a good guess for minimizations
      junk1, h2_eval, common_times, deltaT, deltaPhi = \
         _gwtools.setup_minimization_from_discrete_waveforms(time_sur,h_sur,t_ref,h_ref,t_low_adj,t_up_adj)

      ### (tc,phic)-parameterized waveform function to induce a parameterized norm ###
      def parameterized_waveform(x):
        tc   = x[0]
        phic = x[1]
        times = _gwtools.coordinate_time_shift(common_times,tc)        
        times,hp,hc = self.__call__(q=q,M=M,dist=dist,theta=theta,phi=phic,\
                                  times=times, units=t_ref_units,ell=ell,m=m,fake_neg_modes=fake_neg_modes)
        return hp + 1.0j*hc

    elif speed == 'fast': # build spline interpolant of modes, evaluate the interpolant

      modes_to_evaluate, t_mode, hp_full, hc_full = self.__call__(q=q,M=M,dist=dist,ell=ell,m=m,mode_sum=False,fake_neg_modes=fake_neg_modes)
      h_sphere = _gwtools.h_sphere_builder(modes_to_evaluate, hp_full, hc_full, t_mode)

      hp,hc=h_sphere(t_mode,theta=theta, phi=0.0, z_rot=None, psi_rot=None)
      h1=hp+1.0j*hc

      junk1, h2_eval, common_times, deltaT, deltaPhi = \
          _gwtools.setup_minimization_from_discrete_waveforms(t_mode,h1,t_ref,h_ref,t_low_adj,t_up_adj)
      parameterized_waveform = _gwtools.generate_parameterize_waveform(common_times,h_sphere,'h_sphere',theta)

    else:
      raise ValueError('not coded yet')

    ### solve minimization problem ###
    [guessed_norm,min_norm], opt_solution, hsur_align = \
      _gwtools.minimize_waveform_match(parameterized_waveform,\
                                      h2_eval,_gwtools.euclidean_norm_sqrd,\
                                      [deltaT,-deltaPhi/2.0],'nelder-mead')


    return min_norm, opt_solution, [common_times, hsur_align, h2_eval]


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def h_sphere_builder(self, q, M=None,dist=None,ell=None,m=None):
    """Returns a function for evaluations of h(t,theta,phi;q,M,d) which include 
       all available modes. 

       This new function h(t,theta,phi;q,M,d)
       can be evaluated for rotations about z-axis and at any set of 
       points on the sphere. modes_to_evalute are also returned"""

    modes_to_evaluate, t_mode, hp_full, hc_full = \
      self(q=q, M=M, dist=dist, mode_sum=False,ell=ell,m=m)

    h_sphere = _gwtools.h_sphere_builder(modes_to_evaluate, hp_full, hc_full, t_mode)
    
    return h_sphere, modes_to_evaluate,


  #### below here are "private" member functions ###
  # These routine's carry out inner workings of multimode surrogate
  # class (such as memory allocation)

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def _allocate_output_array(self, times, num_modes, mode_sum):
    """ allocate memory for result of hp, hc.

    Input
    =====
    times     --- array of time samples. None if using default
    num_modes --- number of harmonic modes (cols). set to 1 if summation over modes
    mode_sum  --- whether or not modes will be summed over (see code for why necessary)"""


    # Determine the number of time samples #
    if (times is not None):
      sample_size = times.shape[0]
    else:
      sample_size = self.time_grid().shape[0]

    # TODO: should the dtype be complex?
    if(num_modes==1): # return as vector instead of array
      hp_full = np.zeros(sample_size)
      hc_full = np.zeros(sample_size)
    else:
      hp_full = np.zeros((sample_size,num_modes))
      hc_full = np.zeros((sample_size,num_modes))

    return hp_full, hc_full


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def _generate_minus_m_mode(self,hp_mode,hc_mode,ell,m):
    """ For m>0 positive modes hp_mode,hc_mode use h(l,-m) = (-1)^l h(l,m)^*
        to compute the m<0 mode. 

  See Eq. 78 of Kidder,Physical Review D 77, 044016 (2008), arXiv:0710.0614v1 [gr-qc]."""

    if (m<=0):
      raise ValueError('m must be nonnegative. m<0 will be generated for you from the m>0 mode.')
    else:
      hp_mode =   np.power(-1,ell) * hp_mode
      hc_mode = - np.power(-1,ell) * hc_mode

    return hp_mode, hc_mode


  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def _extend_mode_list_minus_m(self,mode_list):
    """ from list of [(ell,m)] pairs return a new list which includes m<0 too."""

    positive_modes = list(mode_list)
    for ell,m in positive_modes:
      if m>0:
        mode_list.append((ell,-m))
      if m<0:
        raise ValueError('your list already has negative modes!')

    return mode_list


####################################################
# TODO: loop over all data defined in IO class
def CompareSingleModeSurrogate(sur1,sur2):
  """ Compare data defining two surrogates"""

  agrees = []
  different = []
  no_check = []

  for key in sur1.__dict__.keys(): # Py2 list, Py3 iterator

    result = "Checking attribute %s... "%str(key)

    # surrogate data (ie numbers or array of numbers) 
    if key in ['B','V','R','fitparams_phase','fitparams_amp',\
               'fitparams_norm','greedy_points','eim_indices',\
               'time_info','fit_interval','tmin','tmax',\
               'modeled_data','fits_required','dt','times',\
               'fit_min','fit_max']:

      if np.max(np.abs(sur1.__dict__[key] - sur2.__dict__[key])) != 0:
        different.append(key)
      else:
        agrees.append(key)

    # surrogate tags (ie strings)
    elif key in ['fit_type_phase','fit_type_amp','fit_type_norm',\
                 'parameterization','affine_map','surrogate_mode_type',
                 't_units','surrogate_units','norms']:

      if sur1.__dict__[key] == sur2.__dict__[key]:
        agrees.append(key)
      else:
        different.append(key)

    else:
      no_check.append(key)

  print("Agrees:")
  print(agrees)
  print("Different:")
  print(different)
  print("Did not check:")
  print(no_check)










class SurrogateEvaluator(object):
    """
    Class to load and evaluate generic surrogate models.
    Each derived class should do the following:
        1. Choose domain_type as 'Time' or 'Frequency'.
        2. Set keywords for model, see
            self._check_keywords_and_set_defaults.default_keywords for allowed
            keywords.
        3. define _load_dimless_surrogate(), this should return dimensionless
            domain and modes.
        4. define soft_param_lims and hard_param_lims, the limits for
            parameters beyond which warnings/errors are raised.
    See NRHybSur3dq8 for an example.
    """

    def __init__(self, name, domain_type, keywords, soft_param_lims, \
        hard_param_lims):
        """
        name:           Name of the surrogate
        domain_type:    'Time' or 'Frequency'
        keywords:       keywords for this model. For allowed keys see
                        self._check_keywords_and_set_defaults.default_keywords.
                        If keywords['Precessing'] = False, will automatically
                        determine the m<0 modes from the m>0 modes.
        soft_param_lims: Parameter bounds beyond which a warning is raised.
        hard_param_lims: Parameter bounds beyond which an error is raised.
                         Same order and len as x in the __call__ function.
                         Each element is a [minVal, maxVal] pair.
                         Setting soft_param_lims/hard_param_lims to None will
                         skip that particular check.
        """
        self.name = name

        # load the dimensionless surrogate
        self._sur_dimless = self._load_dimless_surrogate()

        self.param_space = self._sur_dimless.param_space
        self.domain      = self._sur_dimless.domain
        self.mode_list   = self._sur_dimless.mode_list

        self._domain_type = domain_type
        if self._domain_type not in ['Time', 'Frequency']:
            raise Exception('Invalid domain_type.')

        # Get some useful keywords, set missing keywords to default values
        self.keywords = keywords
        self._check_keywords_and_set_defaults()

        self.soft_param_lims = soft_param_lims
        self.hard_param_lims = hard_param_lims

        print 'Loaded %s model'%self.name


    def _load_dimless_surrogate(self):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, handles the loading of the dimensionless surrogate.
        This should return the loaded surrogate.
        The loaded surrogate should have a __call__ function that returns the
        dimensionless time/frequency array and dimensionless waveform modes.
        This __call__ function should take the same inputs as
        self._call_dimless_modes
        """
        raise NotImplementedError("Please override me.")


    def _check_keywords_and_set_defaults(self):
        """ Does some sanity checks on self.keywords.
            If any of the default_keywords are not specified, updates
            self.keywords to have these default values.
        """
        default_keywords = {
            'Precessing': False,
            'Eccentric': False,
            'Hybridized': False,
            'nonGR': False,     # We will get there
            }

        # Sanity checks
        if type(self.keywords) != dict:
            raise Exception("Invalid type for self.keywords")
        for key in self.keywords.keys():
            if type(self.keywords[key]) != bool:
                raise Exception("Invalid type for key=%s in self.keywords"%key)
            if key not in default_keywords.keys():
                raise Exception('Invalid key %s in self.keywords'%(key))

        # set to default if keword not specified
        for key in default_keywords:
            if key not in self.keywords.keys():
                self.keywords[key] = default_keywords[key]


    def _check_param_limits(self, x):
        """ Checks that x is within allowed range of paramters.
            Raises a warning if outside self.soft_param_lims and
            raises an error if outside self.hard_param_lims.
            If these are None, skips the checks.
        """
        if len(x) != len(self.soft_param_lims):
            raise Exception("Expected x to be of len=%d"\
                %len(self.soft_param_lims))

        if self.hard_param_lims is not None:
            raise_hard_error = False
            for i in range(len(x)):
                if x[i] < self.hard_param_lims[i][0] \
                    or x[i] > self.hard_param_lims[i][1]:
                    raise_hard_error = True

            if raise_hard_error:
                raise Exception('Parameters x are outside allowed range.')

        if self.soft_param_lims is not None:
            raise_soft_warning = False
            for i in range(len(x)):
                if x[i] < self.soft_param_lims[i][0] \
                    or x[i] > self.soft_param_lims[i][1]:
                    raise_soft_warning = True

            if raise_soft_warning:
                warnings.warn('Parameters x are outside training range.')


    def _call_dimless_modes(self, x, fM_low=None, fM_ref=None,
        dtM=None, timesM=None, dfM=None, freqsM=None, mode_list=None,
        par_dict=None):
        """ Evaluates the surrogate modes in dimensionless units.
        """

        return self._sur_dimless(x, fM_low=fM_low, fM_ref=fM_ref, dtM=dtM,
            timesM=timesM, dfM=dfM, freqsM=freqsM, mode_list=mode_list,
            par_dict=par_dict)


    def _mode_sum(self, h_modes, theta, phi, fake_neg_modes=False):
        """ Sums over h_modes at a given theta, phi.
            If fake_neg_modes = True, deduces m<0 modes from m>0 modes.
            If fake_neg_modes = True, m<0 modes should not be in h_modes.
        """
        h = 0.
        for (ell, m), h_mode in h_modes.items(): # inefficient in py2
            h += _sYlm(-2, ell, m, theta, phi) * h_mode
            if fake_neg_modes:
                if m > 0:
                    h += _sYlm(-2, ell, -m, theta, phi) \
                        * (-1)**ell * h_mode.conjugate()
                elif m < 0:
                    # Looks like this m<0 mode exits, we should be using that.
                    raise Exception('Expected only m>0 modes.')
        return h


    def __call__(self, x, M=None, dist_mpc=None, f_low=None, t_ref=None,
        f_ref=None, dt=None, df=None, times=None, freqs=None, mode_list=None,
        inclination=None, phi_ref=None, par_dict=None, units='dimensionless'):
        """
    INPUT
    =====
    x :         Array of binary parameter values EXCLUDING total mass M.
                This depends on the particular surrogate model.
                Examples:
                    For NRHybSur3dq8, x=[q,chi1z,chi2z].

    M, dist_mpc: Either specify both M and dist_mpc or neither.
        M        :  Total mass (solar masses). Default: None.
        dist_mpc :  Distance to binary system (MegaParsecs). Default: None.

    f_low :     Instantaneous initial frequency of the (2, 2) mode. If None,
                the entire waveform is returned. Default: None.

    f_ref:      Reference frequency used to set the reference epoch at which
                the frame is aligned and the spins are specified. The frame is
                aligned at the reference epoch as follows:
                    The orbital angular momentum points towards the +ve z-axis.
                    The separation vector from the smaller BH to the larger
                    BH points towards the +ve x-axis.
                For time domain models, f_ref is used to detemine a t_ref, such
                that the frequency of the (2, 2) mode equals f_ref at t=t_ref.
                Default: If f_low is given, f_ref = f_low. If f_low is None,
                f_ref is set to the initial frequency (the first index).

    dt, df :    Time/Frequency step size, specify at most one of dt/df,
                depending on whether the surrogate is a time/frequency domain
                surrogate. If None, the internal domain of the surrogate is
                used, which can be nonuniformly sampled. Default None.
                Do not specify times/freqs if using dt/df.

    times, freqs:     
                Array of time/frequency samples at which to evaluate the
                waveform, depending on whether the surrogate is a 
                time/frequency domain surrogate. Default None.
                Do not specify dt/df if using times/freqs.

    mode_list : A list of (l, m) modes to be evaluated. If None, evaluates all
                available modes. Default: None.

    inclination, phi_ref :  Either specify both or neither.
                Evaluate the waveform at this location on the sphere by summing
                over the modes given in the 'mode_list' argument. For
                nonprecessing systems the m<0 modes are automatically deduced
                from the m>0 modes. To see if a model is precessing check
                self.keywords.
                If both inclination and phi_ref are None, the mode data is
                returned as a dictionary. If specified, the complex strain
                (h = hplus -i hcross) evaluated at (inclination, phi_ref) on
                the sky is returned. Defaults: None.

    par_dict:   A dictionary containing any additional parameters needed for a
                particular surrogate model. Default: None.

    units:      'dimensionless' or 'mks'. Default: 'dimensionless'.
                If 'dimensionless': Any of f_low, f_ref, dt, df, times and
                    freqs, if specified, must be in dimensionless units. That
                    is, dt/times should be in units of M, while f_ref, f_low
                    and df/freqs should be in units of cycles/M. 
                    M and dist_mpc must be None. The waveform and domain are
                    returned as dimensionless quantities as well.
                If 'mks': Any of f_low, f_ref, dt, df, times and freqs, if
                    specified, must be in MKS units. That is, dt/times should
                    be in seconds, while f_ref, f_low and df/freqs should be
                    in Hz. M and dist_mpc must be specified. The waveform and
                    domain are returned in MKS units as well.

    RETURNS
    =====
    if times/freqs is given:
        h
    else:
        domain, h

    domain :    Array of time/frequency samples, depending on whether the
                surrogate is a time/frequency domain model. For time domain
                models the time is set to 0 at the peak of the waveform.

    h :         Waveform. If inclination/phi_ref are specified, returns
                complex strain h = hplus -i hcross, evaluated at
                (inclination, phi_ref) on the sky.
                Else, h is a dictionary of available modes with (l, m) tuples
                as keys.
        """

        # Sanity checks
        if (inclination is None) ^ (phi_ref is None):
            raise ValueError("Either specify both inclination and phi_ref,"
                " or neither")

        if (M is None) ^ (dist_mpc is None):
            raise ValueError("Either specify both M and dist_mpc, or neither")

        if (M is not None) ^ (units == 'mks'):
            raise ValueError("M/dist_mpc must be specified if and only if"
                " units='mks'")

        if (dt is not None) and (self._domain_type != 'Time'):
            raise ValueError("%s is not a Time domain model, cannot specify"
                " dt"%self.name)

        if (times is not None) and (self._domain_type != 'Time'):
            raise ValueError("%s is not a Time domain model, cannot specify"
                " times"%self.name)

        if (df is not None) and (self._domain_type != 'Frequency'):
            raise ValueError("%s is not a Frequency domain model, cannot"
                " specify df"%self.name)

        if (freqs is not None) and (self._domain_type != 'Frequency'):
            raise ValueError("%s is not a Frequency domain model, cannot"
                " specify freqs"%self.name)

        if (dt is not None) and (times is not None):
            raise ValueError("Cannot specify both dt and times.")

        if (df is not None) and (freqs is not None):
            raise ValueError("Cannot specify both df and freqs.")

        if (f_ref is not None) and (f_low is not None) and (f_ref < f_low):
            raise ValueError("f_ref cannot be lower than f_low.")

        # Warn/Exit if extrapolating
        self._check_param_limits(x)


        # Get scalings from dimensionless units to mks units
        if units == 'dimensionless':
            amp_scale = 1.0
            t_scale = 1.0
        elif units == 'mks':
            amp_scale = \
                M*_gwtools.Msuninsec*_gwtools.c/(1e6*dist_mpc*_gwtools.PC_SI)
            t_scale = _gwtools.Msuninsec * M
        else:
            raise Exception('Invalid units')

        # If f_ref is not given, we set it to f_low. If f_low is also None,
        # f_ref will correspond to the initial frequency.
        if f_ref is None:
            f_ref = f_low

        # Get dimensionless step size or times/freqs and reference time/freq
        dtM = None if dt is None else dt/t_scale
        timesM = None if times is None else times/t_scale
        dfM = None if df is None else df*t_scale
        freqsM = None if freqs is None else freqs*t_scale
        fM_ref = None if f_ref is None else f_ref*t_scale

        # Get waveform modes and domain in dimensionless units
        fM_low = None if f_low is None else f_low*t_scale
        data = self._call_dimless_modes(x, fM_low=fM_low,
            fM_ref=fM_ref, dtM=dtM, timesM=timesM, dfM=dfM, freqsM=freqsM,
            mode_list=mode_list, par_dict=par_dict)
        if len(data) == 2:
            domain, h = data
        else:
            h = data
            domain = None       # Assuming times/freqs were specified.

        # sum over modes to get complex strain if inclination/phi_ref are given
        if inclination is not None:
            # For nonprecessing systems get the m<0 modes from the m>0 modes.
            fake_neg_modes = not self.keywords['Precessing']
            theta = inclination
            #FIXME document this better once a consensus is reached.
            phi = -phi_ref + np.pi/2        # LAL convention
            h = self._mode_sum(h, theta, phi, fake_neg_modes=fake_neg_modes)

        if domain is not None:
            # Rescale domain to physical units
            if self._domain_type == 'Time':
                domain = domain*t_scale
            elif self._domain_type == 'Frequency':
                domain = domain/t_scale
            else:
                raise Exception('Invalid _domain_type.')

        # Rescale waveform to physical units
        if type(h) == dict:
            h.update((x, y*amp_scale) for x, y in h.items())
        else:
            h = h*amp_scale

        if domain is None:
            return h
        else:
            return domain, h




class NRHybSur3dq8(SurrogateEvaluator):
    """
A class for the NRHybSur3dq8 surrogate model presented in Varma et al. 2018,
in prep.

Evaluates gravitational waveforms generated by aligned-spin binary black hole
systems. This model was built using numerical relativity (NR) waveforms that
have been hybridized using post-Newtonian (PN) and effective one body (EOB)
waveforms.

This model includes the following spin-weighted spherical harmonic modes:
(2,2), (2,1), (2,0), (3,3), (3,2), (3,1), (3,0), (4,4) (4,3), (4,2) and (5,5).
The m<0 modes are deduced from the m>0 modes.

The parameter space of validity is:
q \in [1, 10] and chi1z/chi2z \in [-1, 1],
where q is the mass ratio and chi1z/chi2z are the spins of the larger/smaller
BH, respectively, in the direction of orbital angular momentum.

The surrogate has been trained in the range
q \in [1, 8] and chi1z/chi2z \in [-0.8, 0.8], but produces reasonable waveforms
in the above range and has been tested against existing NR waveforms in that
range.

See the __call__ method on how to evaluate waveforms.
In the __call__ method, x must have format x = [q, chi1z, chi2z].
    """

    def __init__(self, h5filename):
        self.h5filename = h5filename
        domain_type = 'Time'
        keywords = {
            'Precessing': False,
            'Eccentric': False,
            'Hybridized': True,
            }
        # soft_lims -> raise warning when outside lims
        # hard_lim -> raise error when outside lims
        # Same order as x in the call function. Each element is
        # a [minVal, maxVal] pair.
        soft_param_lims = [[0.99, 8.01], [-0.801, 0.801], [-0.801, 0.801]]
        hard_param_lims = [[0.99, 10.01], [-1, 1], [-1, 1]]
        super(NRHybSur3dq8, self).__init__(self.__class__.__name__, \
            domain_type, keywords, soft_param_lims, hard_param_lims)

    def _load_dimless_surrogate(self):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, handles the loading of the dimensionless surrogate.
        This should return the loaded surrogate.
        The loaded surrogate should have a __call__ function that returns the
        dimensionless time/frequency array and dimensionless waveform modes.
        This __call__ function should take the same inputs as
        self._call_dimless_modes.
        """
        sur = new_surrogate.AlignedSpinCoOrbitalFrameSurrogate()
        sur.load(self.h5filename)
        return sur



#### for each model in the catalog (name or h5 file), associate class to load 
#### NOTE: other classes maybe usable too, these just constitute
####       the default cases suitable for most people
SURROGATE_CLASSES = {
    "NRHybSur3dq8": NRHybSur3dq8,
#    "SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0.h5":EvaluateSurrogate # model SpEC_q1_10_NoSpin
        }

# TODO: would this be better off as a function as opposed to a class?
class LoadSurrogate(object):
    """
    A holder class for any SurrogateEvaluator class.
    This is essentially only to let us know what class to
    initialize when loading from an h5 file.
    """

    #NOTE: __init__ is never called for LoadSurrogate
    def __new__(self, surrogate_name):
        if surrogate_name.endswith('.h5'):
            # If h5 file is given, use that directly. But get the
            # surrogate_name used to pick from SURROGATE_CLASSES from the
            # filename
            surrogate_h5file = surrogate_name
            surrogate_name = os.path.basename(surrogate_h5file)
            surrogate_name = surrogate_name.split('.h5')[0]
        else:
            # If not, look for surrogate data in surrogate download_path
            surrogate_h5file = '%s/%s.h5'%(catalog.download_path(), \
                surrogate_name)
            if not os.path.isfile(surrogate_h5file):
                raise Exception("Surrogate data not found. Do"
                    " gwsurrogate.catalog.pull('%s')"%surrogate_name)

        if surrogate_name not in SURROGATE_CLASSES.keys():
            raise Exception('Invalid surrogate : %s'%surrogate_name)
        else:
            return SURROGATE_CLASSES[surrogate_name](surrogate_h5file)

