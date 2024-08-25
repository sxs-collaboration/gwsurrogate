""" Gravitational Wave Surrogate classes for text and hdf5 files"""

from __future__ import division # for python 2

__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@umassd.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Jonathan Blackman, Scott Field, Chad Galley, Vijay Varma, Kevin Barkett"
__version__ = "1.1.6"
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
from gwtools.harmonics import sYlm as _sYlm
from gwtools import plot_pretty as _plot_pretty
from gwtools import gwtools as _gwtools # from the package gwtools, import the module gwtools (gwtools.py)....
from gwtools import gwutils as _gwutils
from .parametric_funcs import function_dict as my_funcs
from .surrogateIO import H5Surrogate as _H5Surrogate
from .surrogateIO import TextSurrogateRead as _TextSurrogateRead
from .surrogateIO import TextSurrogateWrite as _TextSurrogateWrite
from .surrogateIO import BHPTNRCalibValues as _BHPTNRCalibValues
from gwsurrogate.new.surrogate import ParamDim, ParamSpace

import warnings
import os
import copy

from .new import surrogate as new_surrogate
from .new import precessing_surrogate
from . import catalog
from .catalog import get_modelID_from_filename

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
    
    # For models that include the NR calibration info as one of the keys of the h5 file,
    # we need to call the following class
    # in future, when more BHPTNRSurrogates are available, we can replace the if statement
    # to look for the phrase "BHPTNRSur"
    if(self.surrogateID == 'BHPTNRSur1dq1e4'):
      self.nrcalib = _BHPTNRCalibValues(file=path.filename)
    else:
      self.nrcalib = None
    
    # Interpolate columns of the empirical interpolant operator, B, using cubic spline
    if self.surrogate_mode_type  == 'waveform_basis':
      self.reB_spline_params = [_splrep(self.times, self.B[:,jj].real, k=deg) for jj in range(self.B.shape[1])]
      self.imB_spline_params = [_splrep(self.times, self.B[:,jj].imag, k=deg) for jj in range(self.B.shape[1])]
    elif self.surrogate_mode_type in ['amp_phase_basis', 'coorb_waveform_basis']:
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



    # Subsequent functions (e.g. code that checks evaluation point within training) assumes this
    assert(q>=1)
    
    # surrogate evaluations assumed dimensionless, physical modes are found from scalings
    if self.surrogate_units != 'dimensionless':
      raise ValueError('surrogate units is not supported')

    if (units != 'dimensionless') and (units != 'mks'):
      raise ValueError('units is not supported')

    if (M is not None) and (dist is not None) and (times is not None) and (units != 'mks'):
    	raise ValueError('passing values of M, dist, and times suggest mks units should be used!')

    if self.surrogate_mode_type == 'coorb_waveform_basis' and singlemode_call:
      msg = 'directly calling a coorb_waveform_basis surrogate will return a waveform in the co-orbital frame.\n'
      msg += 'Please use EvaluateSurrogate to evaluate co-orbital frame surrogates.\n'
      msg += 'Use EvaluateSurrogate call method to evaluate in the inertial frame.\n'
      msg += 'Use EvaluateSurrogate.evaluate_single_mode to evaluate in the co-orbital frame.\n' 
      raise ValueError(msg)


    # For models with NR calibration information, we need to find the calibration values
    # at a given value of parameter space. These calibration parameters will scale the 
    # time and amplitude. Currently, these models use calibration:
    # BHPTNRSur1dq1e4, EMRISur1dq1e4
    if(self.surrogateID == 'BHPTNRSur1dq1e4'):
      alpha_nr_calibration, beta_nr_calibration = self.compute_BHPT_calibration_params(q)
    
    if(self.surrogateID == 'EMRISur1dq1e4'):
      # see Eq 4 of https://arxiv.org/abs/1910.10473
      # if alpha = 1 we have the "raw" waveform computed from
      # point-particle black hole perturbation theory
      # alpha rescales all modes in the same way in time and amplitude
      nu         = q/(1.+q)**2.
      alpha_emri = 1.0-1.352854*nu-1.223006*nu*nu+8.601968*nu*nu*nu-46.74562*nu*nu*nu*nu
    else:
      alpha_emri = None


    ### if (M,distance) provided, a physical mode in mks units is returned ###
    # TODO: use gwtools' routine geo_to_SI to do this conversion as a 1 liner
    if( M is not None and dist is not None):
      amp0    = ((M * _gwtools.MSUN_SI ) / (1.e6*dist*_gwtools.PC_SI )) * ( _gwtools.G / np.power(_gwtools.c,2.0) )
      t_scale = _gwtools.Msuninsec * M
    else:
      amp0    = 1.0
      t_scale = 1.0

    # any model-specific amplitude scalings should go here
    if(self.surrogateID == 'EMRISur1dq1e4'):
      amp0 = alpha_emri * amp0
    if(self.surrogateID == 'BHPTNRSur1dq1e4'):
      amp0 = alpha_nr_calibration * amp0


    ### evaluation times t: input times or times at which surrogate was built ###
    if (times is not None):
      t = times
    else:
      t = self.time() # shallow copy of self.times
      if self.surrogateID == 'EMRISur1dq1e4':
        t = t * alpha_emri # this will deep copy, preserving data in self.times
      elif self.surrogateID == 'BHPTNRSur1dq1e4':
        t = t * beta_nr_calibration

    ### if input times are dimensionless, convert to MKS if a physical surrogate is requested ###
    if units == 'dimensionless':
      t = t_scale * t  # this will deep copy, preserving data in self.times

    # because times is passed to _h_sur, it must be dimensionless form t/M
    if times is not None and units == 'mks':
      times = times / t_scale

    # input time grid needs to be mapped back to "raw EMRI" time grid before evaluation
    if times is not None and self.surrogateID == 'EMRISur1dq1e4':
      times = times / alpha_emri
    if times is not None and self.surrogateID == 'BHPTNRSur1dq1e4':
      times = times / beta_nr_calibration

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

    # different models were built using different conventions of hlm and hp \pm i hx
    if self.surrogateID == 'EMRISur1dq1e4':
      return t, hp, -hc
    else:
  	  return t, hp, hc

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def compute_BHPT_calibration_params(self, q):
    """
    Computes alpha and beta calibration parameters for a given mode and mass ratio.

    These parameters are used to calibrate point-particle perturbation theory waveforms to NR.

    Please see: https://arxiv.org/abs/2204.01972

    """

    # evaluate alpha and beta from polynomials

    if self.mode_ell<=5:
      # alpha calibration parameters from ell=m data applied to ell!=m
      alpha_coeffs_mode = self.nrcalib.alpha_coeffs[(float(self.mode_ell),float(self.mode_ell))]
      alpha = my_funcs["BHPT_nrcalib_functional_form"](1/q, alpha_coeffs_mode[0], alpha_coeffs_mode[1],\
                                                      alpha_coeffs_mode[2], alpha_coeffs_mode[3])
    else:
      alpha = 1.0

    # beta parameters independent of ell,m
    beta_coeffs = self.nrcalib.beta_coeffs
    beta = my_funcs["BHPT_nrcalib_functional_form"](1/q, beta_coeffs[0], beta_coeffs[1],\
                                                     beta_coeffs[2], beta_coeffs[3])
    return alpha, beta

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
  def _coorb_re_eval(self, x_0):
    """Evaluate set of coorbital real part of waveform fits at x_0, where x_0 is the mapped parameter value.

       WARNING: this function should NEVER be called from outside the class."""

    if self.fit_type_amp == 'fast_spline_real':
      return self.amp_fit_func(self.fitparams_re, x_0)
    else:
      return np.array([ self.re_fit_func(self.fitparams_re[jj,:], x_0) for jj in range(self.fitparams_re.shape[0]) ])
  
  def _coorb_im_eval(self, x_0):
    """Evaluate set of coorbital imag part of waveform fits at x_0, where x_0 is the mapped parameter value.

       WARNING: this function should NEVER be called from outside the class."""

    if self.fit_type_amp == 'fast_spline_real':
      return self.amp_fit_func(self.fitparams_im, x_0)
    else:
      return np.array([ self.im_fit_func(self.fitparams_im[jj,:], x_0) for jj in range(self.fitparams_im.shape[0]) ])

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
    if self.surrogate_mode_type  == 'coorb_waveform_basis':
      re_eval = self._coorb_re_eval(x_0)
      im_eval = self._coorb_im_eval(x_0)
    else:
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
    elif self.surrogate_mode_type  == 'coorb_waveform_basis':
      if self.fit_type_amp == 'fast_spline_real':
        raise ValueError("invalid combination")
      return re_eval, im_eval, nrm_eval
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

    elif self.surrogate_mode_type  == 'coorb_waveform_basis':

      re_eval, im_eval, nrm_eval = self._eim_coeffs(x, 'coorb_waveform_basis')

      if times is None:
        sur_Re = np.dot(self.B_1, re_eval)
        sur_Im = np.dot(self.B_2, im_eval)
      else:
        sur_Re = np.dot(self.resample_B_1(times), re_eval)
        sur_Im = np.dot(self.resample_B_2(times), im_eval)
   
      surrogate = nrm_eval * (sur_Re + 1j*sur_Im)
    

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
            ell = int(splitkk[0][1:])
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
            ell = int(splitkk[0][1:])
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
        will explicitly check that m<0 do not exist in the data file and (ii) m<0 modes
        are inferred from m>0 modes. If set to false no symmetry is assumed -- typical
        of precessing models. When False, fake_neg_modes must be false."""

    # obtain the surrogate ID name from the datafile
    # it is important for some of the if statements written later on
    surrogateID = get_modelID_from_filename(path)
    if len(surrogateID) == 0 or len(surrogateID) > 1:
      self.surrogateID = None
      print("\n>>> Warning: No surrogate ID found. Could not deduce ID from file")
    else:
      self.surrogateID = surrogateID[0]

    if self.surrogateID == 'BHPTNRSur1dq1e4':
      msg = "co-orbital surrogate models must load the 22 mode data as other modes depend on it" 
      assert (ell_m is None or (2,2) in ell_m), msg
 
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
    pd = ParamDim(name='unknown parameter',
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
    # For models where the higher modes have been modeled in the coorbital frame, 
    # it is necessary to have the (2,2) mode as the first index of modes_to_evaluate
    # otherwise, we do not have the information regarding the orbital phase
    # crucial to transform the higher modes from coorbital frame to inertial frame
    # this line of code guarantees that if mode_sum is requested, (2,2) mode
    # should always be in the modes_to_evaluate list
    if self.surrogateID=="BHPTNRSur1dq1e4" and mode_sum:
      assert (2,2) in modes_to_evaluate, "Must include 22 in mode_sum for the BHPTNRSur1dq1e4 model"

    if mode_sum and (theta is None and phi is None) and len(modes_to_evaluate)!=1:
      raise ValueError('Trying to sum modes without theta and phi is a strange idea')

    # For models where the higher modes have been modeled in the coorbital frame, 
    # it is necessary to have the (2,2) mode as the first index of modes_to_evaluate
    # otherwise, we do not have the information regarding the orbital phase
    # crucial to transform the higher modes from coorbital frame to inertial frame
    # this line of code guarantees that 
    #   (1) (2,2) is always included in the modes to evaluate
    #   (2) (2,2) is the first mode in modes_to_evaluate
    # The code will return the (2,2) mode along with the other modes requested
    if self.surrogateID=='BHPTNRSur1dq1e4':
      modes_to_evaluate = self.add_l2m2_mode_if_not_in_modelist(modes_to_evaluate)
    
    ### if mode_sum false, return modes in a sensible way ###
    if not mode_sum:
      # For models where the higher modes have been modeled in the coorbital frame, 
      # it is necessary to have the [2,2] mode as the first index of modes_to_evaluate
      # otherwise, we do not have the information regarding the orbital phase
      # crucial to transform the higher modes from coorbital frame to inertial frame
      # this line of code guarantees that [2,2] always comes first by not sorting
      # as sorting makes [l,-m] the first mode in the array
      if self.surrogateID!='BHPTNRSur1dq1e4':
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

      ### if the mode is modeled, evaluate it. Otherwise its zero ###
      is_modeled = (ell,m) in modeled_modes
      neg_modeled = (ell,-m) in modeled_modes
      if is_modeled or (neg_modeled and fake_neg_modes):

        # if model is BHPTNRSur1dq1e4 and mode not 22, hp/hc are in the coorbital frame
        if is_modeled:
          t_mode, hp_mode, hc_mode = self.evaluate_single_mode(q,M,dist,f_low,times,units,ell,m)
        else: # then we must have neg_modeled=True and fake_neg_modes=True
          t_mode, hp_mode, hc_mode = self.evaluate_single_mode_by_symmetry(q,M,dist,f_low,times,units,ell,m)

        # BHPTNRSur1dq1e4 models the real and imag parts of the higher order modes 
        # in the coorbital frame. We have to make sure we apply a frame transformation
        # for this surrogate modes
        if self.surrogateID=='BHPTNRSur1dq1e4':
          if [ell,m] in [[2,2],[2,-2]]:
            if [ell,m]==[2,2]:
               orbital_phase = 0.5 * _gwtools.phase(hp_mode + 1j * hc_mode)
          else:
               hp_mode, hc_mode = self.coorbital_to_inertial(hp_mode, hc_mode, m, orbital_phase)

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
  def coorbital_to_inertial(self, coorb_re, coorb_im, m, orbital_phase):
    """ Takes the real and imaginary part of the waveform and
    combine them to obtain the coorbital frame waveform;
    then transform the wf into the inertial frame"""
    
    full_wf = (coorb_re+1j*coorb_im)*np.exp(1j*m*np.array(orbital_phase))
    return full_wf.real, full_wf.imag

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  def add_l2m2_mode_if_not_in_modelist(self, list_of_modes):
    """
    Adds the (2,2) mode if it is not in ell, m list. This is 
    required for some models which evaluates coorbital frame waveform
    for the higher modes (2,2) mode info is required here.

    This routine also ensures (2,2) is the front of the list
    """
    if (2,2) not in list_of_modes:
      list_of_modes.insert(0, (2,2))  # prepend
    else:
      where_is_22 = list_of_modes.index( (2,2) )
      list_of_modes.insert(0, list_of_modes.pop(where_is_22) ) # move (2,2) to the front

    return list_of_modes

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
  # TODO: this routine will evaluate m>0 to get the m<0 case!!! It would be much better
  #       to simply use m>0 modes that were already evaluated (could be ~2x faster) 
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
        3. define _load_dimless_surrogate(), this should return an object that
            returns dimensionless domain, modes and dynamics.
        4. define _get_intrinsic_parameters(), this should put all intrinsic
            parameters into a single array.
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
                         Should be in format [qMax, chimax]
                         Setting soft_param_lims/hard_param_lims to None will
                         skip that particular check.
        """
        self.name = name

        # load the dimensionless surrogate
        self._sur_dimless = self._load_dimless_surrogate()

        self._domain_type = domain_type
        if self._domain_type not in ['Time', 'Frequency']:
            raise Exception('Invalid domain_type.')

        # Get some useful keywords, set missing keywords to default values
        self.keywords = keywords
        self._check_keywords_and_set_defaults()

        self.soft_param_lims = soft_param_lims
        self.hard_param_lims = hard_param_lims

        print('Loaded %s model'%self.name)


    def _load_dimless_surrogate(self):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, handles the loading of the dimensionless surrogate.
        This should return the loaded surrogate.
        The loaded surrogate should have a __call__ function that returns the
        dimensionless time/frequency array and dimensionless waveform modes.
        The return value of this functions will be stored as
        self._sur_dimless()
        The __call__ function of self._sur_dimless() should take all inputs
        passed to self._sur_dimless() in the __call__ function of this class.
        See NRHybSur3dq8 for an example.
        """
        raise NotImplementedError("Please override me.")


    def _get_intrinsic_parameters(self, q, chiA0, chiB0, precessing_opts,
            tidal_opts, par_dict):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, puts all intrinsic parameters of the surrogate
        into a single array.
        For example:
            For NRHybSur3dq8: x = [q, chiAz, chiBz].
            For NRSur7dq4: x = [q, chiA, chiB], where chiA/chiB are vectors of
                size 3.
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
            'Tidal': False,
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


    def _check_params(self, q, chiA0, chiB0, precessing_opts, tidal_opts,
            par_dict):
        """ Checks that the parameters are valid.

            Raises a warning if outside self.soft_param_lims and
            raises an error if outside self.hard_param_lims.
            If these are None, skips the checks.

            Also some sanity checks for precessing and tidal models.
        """
        ## Allow violations within this value.
        # Sometimes, chi can be 1+1e-16 due to machine precision limitations,
        # this will ignore such cases
        grace = 1e-14

        chiAmag = np.linalg.norm(chiA0)
        chiBmag = np.linalg.norm(chiB0)

        if not self.keywords['Precessing']:
            if (np.linalg.norm(chiA0[:2]) > grace
                    or np.linalg.norm(chiB0[:2]) > grace):
                raise Exception('Got precessing spins for a nonprecessing '
                    'model')

            if precessing_opts is not None:
                raise Exception('precessing_opts should be None for '
                        'nonprecessing models')


        if self.keywords['Tidal']:
            if (tidal_opts is None) or (('Lambda1' not in tidal_opts.keys())
                    or ('Lambda2' not in tidal_opts.keys())):
                raise Exception('Tidal parameters Lambda1 and Lambda2 should '
                        'be passed through tidal_opts for this model.')
        else:
            if tidal_opts is not None:
                raise Exception('tidal_opts should be None for nontidal '
                        'models')


        # Extrapolation checks
        if self.hard_param_lims is not None:
            qMax = self.hard_param_lims[0]
            chiMax = self.hard_param_lims[1]
            if q > qMax + grace or q < 0.99:
                raise Exception('Mass ratio q=%.4f is outside allowed '
                    'range: 1<=q<=%.4f'%(q, qMax))
            if chiAmag > chiMax + grace:
                raise Exception('Spin magnitude of BhA=%.4f is outside '
                    'allowed range: chi<=%.4f'%(chiAmag, chiMax))
            if chiBmag > chiMax + grace:
                raise Exception('Spin magnitude of BhB=%.4f is outside '
                    'allowed range: chi<=%.4f'%(chiBmag, chiMax))

        if self.soft_param_lims is not None:
            qMax = self.soft_param_lims[0]
            chiMax = self.soft_param_lims[1]
            if q > qMax:
                warnings.warn('Mass ratio q=%.4f is outside training '
                    'range: 1<=q<=%.4f'%(q, qMax))
            if chiAmag > chiMax:
                warnings.warn('Spin magnitude of BhA=%.4f is outside '
                    'training range: chi<=%.4f'%(chiAmag, chiMax))
            if chiBmag > chiMax:
                warnings.warn('Spin magnitude of BhB=%.4f is outside '
                    'training range: chi<=%.4f'%(chiBmag, chiMax))



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


    def __call__(self, q, chiA0, chiB0, M=None, dist_mpc=None, f_low=None,
        f_ref=None, dt=None, df=None, times=None, freqs=None,
        mode_list=None, ellMax=None, inclination=None, phi_ref=0,
        precessing_opts=None, tidal_opts=None, par_dict=None,
        units='dimensionless', skip_param_checks=False,
        taper_end_duration=None):
        """
    INPUT
    =====
    q :         Mass ratio, mA/mB >= 1.
    chiA0:      Dimensionless spin vector of the heavier black hole at
                reference epoch.
    chiB0:      Dimensionless spin vector of the lighter black hole at
                reference epoch.

                This follows the same convention as LAL, where the spin
                components are defined as:
                \chi_z = \chi \cdot \hat{L}, where L is the orbital angular
                    momentum vector at the epoch.
                \chi_x = \chi \cdot \hat{n}, where n = body2 -> body1 is the
                    separation vector at the epoch. body1 is the heavier body.
                \chi_y = \chi \cdot \hat{L \cross n}.
                These spin components are frame-independent as they are
                defined using vector inner products. This is equivalent to
                specifying the spins in the coorbital frame used in the
                surrogate papers.

    M, dist_mpc: Either specify both M and dist_mpc or neither.
        M        :  Total mass (solar masses). Default: None.
        dist_mpc :  Distance to binary system (MegaParsecs). Default: None.

    f_low :     Instantaneous initial frequency of the (2, 2) mode. In
                practice, this is estimated to be twice the initial orbital
                frequency in the coprecessing frame. Note: the coprecessing
                frame is the minimal rotation frame of arXiv:1110.2965.

                f_low should be in cycles/M if units = 'dimensionless',
                should be in Hertz if units = 'mks'.
                If 0, the entire waveform is returned.
                Default: None, must be specified by user.

                NOTE: For some models like NRSur7dq4, f_low=0 is recommended.
                The role of f_low is only to truncate the lower frequencies
                before returning the waveform. Since this model is already
                very short, this truncation is not required. On the other hand,
                f_ref is used to set the reference epoch, and can be freely
                specified.

                WARNING: Using f_low=0 with a small dt (like 0.1M) can lead to
                very expensive evaluation for hybridized surrogates like
                NRHybSur3dq8.

    f_ref:      Frequency used to set the reference epoch at which the
                reference frame is defined and the spins are specified.
                See below for definition of the reference frame.
                Should be in cycles/M if units = 'dimensionless', should be
                in Hertz if units = 'mks'.
                Default: If f_ref is not given, we set f_ref = f_low. If
                f_low is 0, this corresponds to the initial index.

                For time domain models, f_ref is used to determine a t_ref,
                such that the orbital frequency in the coprecessing frame
                equals f_ref/2 at t=t_ref.

    dt, df :    Time/Frequency step size, specify at most one of dt/df,
                depending on whether the surrogate is a time/frequency domain
                surrogate.
                Default: None. If None, the internal domain of the surrogate is
                used, which can be nonuniformly sampled.
                dt (df) Should be in M (cycles/M) if units = 'dimensionless',
                should be in seconds (Hertz) if units = 'mks'. Do not specify
                times/freqs if using dt/df.


    times, freqs:
                Array of time/frequency samples at which to evaluate the
                waveform, depending on whether the surrogate is a
                time/frequency domain surrogate. time (freqs) should be in
                M (cycles/M) if units = 'dimensionless', should be in
                seconds (Hertz) if units = 'mks'. Do not specify dt/df if
                using times/freqs. Default None.

    ellMax:     Maximum ell index for modes to include. All available m
                indicies for each ell will be included automatically. The 
                m<0 modes will automatically be included for nonprecessing
                models. 
                Default: None, in which case all available ells will be
                included. 

    mode_list : A list of (ell, m) modes tuples to be included. Valid only
                for nonprecessing models.

                Example: mode_list = [(2,2),(2,1)].
                Default: None, in which case all available modes are included.

                At most one of ellMax and mode_list can be specified.

                Note: mode_list is allowed only for nonprecessing models; for
                precessing models use ellMax. For precessing systems, all m
                indices of a given ell index mix with each other, so there is
                no clear hierarchy. To get the individual modes just don't
                specify inclination and a dictionary of modes will be returned.

                Note: When the inclination is set, the m<0 modes are 
                automatically included. For example, passing mode_list = [(2,2)] 
                will include the (2,2) and (2,-2) modes in the computation of
                the strain.

                Note: When the inclination is None, the m<0 modes are
                automatically generated.

    inclination : Inclination angle between the orbital angular momentum
                direction at the reference epoch and the line-of-sight to the
                observer. If inclination is None, the mode data is returned
                as a dictionary.
                Default: None.

    phi_ref :   The azimuthal angle on the sky of the source frame following
                the LAL convention.
                Default: 0.

                If inclination/phi_ref are specified, the complex strain (h =
                hplus -i hcross) evaluated at (inclination, pi/2 - phi_ref) on
                the sky of the reference frame is returned. This follows the
                same convention as LAL. See below for definition of the
                reference frame.

    precessing_opts:
                A dictionary containing optional parameters for a precessing
                surrogate model. Default: None.
                Allowed keys are:
                init_orbphase: The orbital phase in the coprecessing frame
                    at the reference epoch.
                    Default: 0, in which case the coorbital frame and
                    coprecessing frame are the same.
                init_quat: The unit quaternion (length 4 vector) giving the
                    rotation from the coprecessing frame to the inertial frame
                    at the reference epoch.
                    Default: None, in which case the coprecessing frame is the
                    same as the inertial frame.
                return_dynamics:
                    Return the frame dynamics and spin evolution along with
                    the waveform. Default: False.
                Example: precessing_opts = {
                                    'init_orbphase': 0,
                                    'init_quat': [1,0,0,0],
                                    'return_dynamics': True
                                    }

    tidal_opts:
                A dictionary containing optional parameters for a tidal
                surrogate model. Default: None.
                Allowed keys are:
                Lambda1: The tidal deformability parameter for the heavier
                    object.
                Lambda2: The tidal deformability parameter for the lighter
                    object.
                Example: tidal_opts = {'Lambda1': 200, 'Lambda2': 300}


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


    skip_param_checks :
                Skip sanity checks for inputs. Use this if you want to
                extrapolate outside allowed range. Default: False.

    taper_end_duration:
                Taper the last TAPER_END_DURATION (M) of a time-domain waveform
                in units of M. For exmple, passing 40 will taper the last 40M.
                When set to None, no taper is applied
                Default: None.

    RETURNS
    =====

    domain, h, dynamics


    domain :    Array of time/frequency samples corresponding to h and
                dynamics, depending on whether the surrogate is a
                time/frequency domain model. This is the same as times/freqs
                if times/freqs are given as an inputs.
                For time domain models the time is set to 0 at the peak of
                the waveform. The time (frequency) values are in M (cycles/M)
                if units = 'dimensionless', they are in seconds (Hertz) if
                units = 'mks'

    h :         The waveform.
                    If inclination is specified, the complex strain (h = hplus
                    -i hcross) evaluated at (inclination, pi/2 - phi_ref) on
                    the sky of the reference frame is returned. This follows
                    the LAL convention, see below for details.  This includes
                    all modes given in the ellMax/mode_list argument. For
                    nonprecessing systems the m<0 modes are automatically
                    deduced from the m>0 modes. To see if a model is precessing
                    check self.keywords.

                    Else, h is a dictionary of available modes with (l, m)
                    tuples as keys. For example, h22 = h[(2,2)].

                    If M and dist_mpc are given, the physical waveform
                    at that distance is returned. Else, it is returned in
                    code units: r*h/M extrapolated to future null-infinity.

    dynamics:   A dict containing the frame dynamics and spin evolution. This
                is None for nonprecessing models. This is also None if
                return_dynamics in precessing_opts is False (Default).

                The dynamics include (L=len(domain)):

                q_copr = dynamics['q_copr']
                    The quaternion representing the coprecessing frame with
                    shape (4, L)
                orbphase = dynamics['orbphase']
                    The orbital phase in the coprecessing frame with length L.
                chiA = dynamics['chiA']
                    The inertial frame chiA with shape (L, 3)
                chiB = dynamics['chiB']
                    The inertial frame chiB with shape (L, 3)


    IMPORTANT NOTES:
    ===============

    The reference frame (or inertial frame) is defined as follows:
        The +ve z-axis is along the orbital angular momentum at the reference
        epoch. The separation vector from the lighter BH to the heavier BH at
        the reference epoch is along the +ve x-axis. The y-axis completes the
        right-handed triad. The reference epoch is set using f_ref.

        Now, if inclination/phi_ref are given, the waveform is evaluated at
        (inclination, pi/2 - phi_ref) in the reference frame. This agrees with
        the LAL convention. See LIGO DCC document T1800226 for the LAL frame
        diagram.
        """

        chiA0 = np.array(chiA0)
        chiB0 = np.array(chiB0)

        # copy dictionary input to avoid modifying them
        precessing_opts = copy.deepcopy(precessing_opts)
        tidal_opts = copy.deepcopy(tidal_opts)
        par_dict = copy.deepcopy(par_dict)

        # Sanity checks
        if not skip_param_checks:

            if (M is None) ^ (dist_mpc is None):
                raise ValueError("Either specify both M and dist_mpc, or "
                        "neither")

            if (M is not None) ^ (units == 'mks'):
                raise ValueError("M/dist_mpc must be specified if and only if"
                    " units='mks'")

            if (dt is not None) and (self._domain_type != 'Time'):
                raise ValueError("%s is not a Time domain model, cannot "
                        "specify dt"%self.name)

            if (times is not None) and (self._domain_type != 'Time'):
                raise ValueError("%s is not a Time domain model, cannot "
                        "specify times"%self.name)

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

            if (f_low is None):
                raise ValueError("f_low must be specified.")

            if (f_ref is not None) and (f_ref < f_low):
                raise ValueError("f_ref cannot be lower than f_low.")

            if (mode_list is not None) and (ellMax is not None):
                raise ValueError("Cannot specify both mode_list and ellMax.")

            if (mode_list is not None) and self.keywords['Precessing']:
                raise ValueError("mode_list is not allowed for precessing "
                        "models, use ellMax instead.")

            if (taper_end_duration is not None) and self._domain_type !='Time':
                raise ValueError("%s is not a Time domain model, cannot taper")

            # more sanity checks including extrapolation checks
            self._check_params(q, chiA0, chiB0, precessing_opts, tidal_opts,
                    par_dict)


        x = self._get_intrinsic_parameters(q, chiA0, chiB0, precessing_opts,
            tidal_opts, par_dict)


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

        # If f_ref is not given, we set it to f_low.
        if f_ref is None:
            f_ref = f_low

        # Get dimensionless step size or times/freqs and reference time/freq
        dtM = None if dt is None else dt/t_scale
        timesM = None if times is None else times/t_scale
        dfM = None if df is None else df*t_scale
        freqsM = None if freqs is None else freqs*t_scale


        # Get waveform modes and domain in dimensionless units
        fM_low = f_low*t_scale
        fM_ref = f_ref*t_scale
        domain, h, dynamics = self._sur_dimless(x, fM_low=fM_low,
            fM_ref=fM_ref, dtM=dtM, timesM=timesM, dfM=dfM,
            freqsM=freqsM, mode_list=mode_list, ellMax=ellMax,
            precessing_opts=precessing_opts, tidal_opts=tidal_opts,
            par_dict=par_dict)

        # taper the last portion of the waveform, regardless of whether or not
        # this corresponds to inspiral, merger, or ringdown.
        if taper_end_duration is not None:
            h_tapered = {}
            for mode, hlm in h.items():
                # NOTE: we use a roll on window [domain[0]-100, domain[0]-50]
                # to trick the window function into not tapering the beginning
                # of h
                h_tapered[mode] = _gwutils.windowWaveform(domain, hlm, \
                    domain[0]-100, domain[0]-50, \
                    domain[-1] - taper_end_duration, domain[-1], \
                    windowType="planck")

            h = h_tapered

        # sum over modes to get complex strain if inclination is given
        if inclination is not None:
            # For nonprecessing systems get the m<0 modes from the m>0 modes.
            fake_neg_modes = not self.keywords['Precessing']

            # Follows the LAL convention (see help text)
            h = self._mode_sum(h, inclination, np.pi/2 - phi_ref,
                    fake_neg_modes=fake_neg_modes)
        else: # if returning modes, check if m<0 modes need to be generated for nonprecessing systems
            if not self.keywords['Precessing']:
                modes = list(h.keys())
                for mode in modes:
                    ell = mode[0]
                    m   = mode[1]
                    if (m > 0) and ( (ell,-m) not in h.keys()):
                        h[(ell,-m)] = (-1)**ell * h[(ell,m)].conjugate()

        # Rescale domain to physical units
        if self._domain_type == 'Time':
            domain *= t_scale
        elif self._domain_type == 'Frequency':
            domain /= t_scale
        else:
            raise Exception('Invalid _domain_type.')

        # Assuming times/freqs were specified, so they must be the same
        # when returning
        if (times is not None):
            # rtol=1e-05, atol=1e-08 were numpy v1.24 defaults. We have hardcoded
            # them here to make the values transparent and guard against
            # future changes to the defaults by numpy developers
            if not np.allclose(domain, times, rtol=1e-05, atol=1e-08):
                raise Exception("times were given as input but returned "
                    "domain somehow does not match.")
        if (freqs is not None):
            if not np.allclose(domain, freqs, rtol=1e-05, atol=1e-08):
                raise Exception("freqs were given as input but returned "
                    "domain somehow does not match.")

        # Rescale waveform to physical units
        if amp_scale != 1:
            if type(h) == dict:
                h.update((x, y*amp_scale) for x, y in h.items())
            else:
                h *= amp_scale

        return domain, h, dynamics




class NRHybSur3dq8(SurrogateEvaluator):
    """
A class for the NRHybSur3dq8 surrogate model presented in Varma et al. 2018,
arxiv:1812.07865.

Evaluates gravitational waveforms generated by aligned-spin binary black hole
systems. This model was built using numerical relativity (NR) waveforms that
have been hybridized using post-Newtonian (PN) and effective one body (EOB)
waveforms.

This model includes the following spin-weighted spherical harmonic modes:
(2,2), (2,1), (2,0), (3,3), (3,2), (3,1), (3,0), (4,4) (4,3), (4,2) and (5,5).
The m<0 modes are deduced from the m>0 modes.

The parameter space of validity is:
q \in [1, 10] and chi1z/chi2z \in [-1, 1],
where q is the mass ratio and chi1z/chi2z are the spins of the heavier/lighter
BH, respectively, in the direction of orbital angular momentum.

The surrogate has been trained in the range
q \in [1, 8] and chi1z/chi2z \in [-0.8, 0.8], but produces reasonable waveforms
in the above range and has been tested against existing NR waveforms in that
range.

See the __call__ method on how to evaluate waveforms.
   """

    def __init__(self, h5filename):
        self.h5filename = h5filename
        domain_type = 'Time'
        keywords = {
            'Precessing': False,
            'Hybridized': True,
            }
        # soft_lims -> raise warning when outside lims
        # hard_lim -> raise error when outside lims
        # Format is [qMax, chiMax].
        soft_param_lims = [8.01, 0.801]
        hard_param_lims = [10.01, 1]
        super(NRHybSur3dq8, self).__init__(self.__class__.__name__, \
            domain_type, keywords, soft_param_lims, hard_param_lims)

    def _load_dimless_surrogate(self):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, handles the loading of the dimensionless surrogate.
        This should return the loaded surrogate.
        The loaded surrogate should have a __call__ function that returns the
        dimensionless time/frequency array and dimensionless waveform modes.
        The return value of this functions will be stored as
        self._sur_dimless()
        The __call__ function of self._sur_dimless() should take all inputs
        passed to self._sur_dimless() in the __call__ function of this class.
        """
        sur = new_surrogate.AlignedSpinCoOrbitalFrameSurrogate()
        sur.load(self.h5filename)
        return sur

    def _get_intrinsic_parameters(self, q, chiA0, chiB0, precessing_opts,
            tidal_opts, par_dict):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, puts all intrinsic parameters of the surrogate
        into a single array.
        For example, for NRHybSur3dq8: x = [q, chiAz, chiBz].
        """
        if par_dict is not None:
            raise ValueError('Expected par_dict to be None.')
        x = [q, chiA0[2], chiB0[2]]
        return x


class NRHybSur3dq8Tidal(SurrogateEvaluator):
    """
A class for the NRHybSur3dq8Tidal model presented in Barkett et al.,
arxiv:xxxx.xxxx #FIXME.

Generates inspiralling gravitational waveforms corresponding to binary neutron
stars/black hole-neutron star systems. This model is based on the aligned-spin
BBH surrogate model of Varma et al. 2018, arxiv:1812.07865. Analytic TaylorT2
PN tidal expressions are then utilized to modify the orbital evolution and
waveform modes.

This model includes the following spin-weighted spherical harmonic modes:
(2,2), (2,1), (2,0), (3,3), (3,2), (3,1), (3,0), (4,4) (4,3), (4,2) and (5,5).
The m<0 modes are deduced from the m>0 modes.

The parameter space of validity is:
q \in [1, 8] and chi1z/chi2z \in [-.7, .7] and lambda1/lambda2 \in [0,10000],
where q is the mass ratio and chi1z/chi2z are the spins of the heavier/lighter
BH, respectively, in the direction of orbital angular momentum, and lambda1/
lambda2 are the dimensionless quadrupolar tidal deformabilities of the
heavier/lighter object, respectively.

The .7 spin restriction is both a theoretical and practical decision.
(i) A .7 spin is an estimate for the breakup speed for NS.
(ii) While the model doesn't allow greater spins if one object is a BH,
that could be allowed. However, with greater spins, the model exhibits
problematic behavior in the waveform at late times as the spin-tidal
crossterms grow significant. This is future work.

See the __call__ method on how to evaluate waveforms.
    """

    def __init__(self, h5filename):
        self.h5filename = h5filename
        domain_type = 'Time'
        keywords = {
            'Tidal': True,
            'Hybridized': True,
            }
        # soft_lims -> raise warning when outside lims
        # hard_lim -> raise error when outside lims
        # Format is [qMax, chiMax].
        soft_param_lims = [8.01, 0.701]
        hard_param_lims = [8.01, 0.701]
        super(NRHybSur3dq8Tidal, self).__init__(self.__class__.__name__, \
            domain_type, keywords, soft_param_lims, hard_param_lims)

    def _load_dimless_surrogate(self):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, handles the loading of the dimensionless surrogate.
        This should return the loaded surrogate.
        The loaded surrogate should have a __call__ function that returns the
        dimensionless time/frequency array and dimensionless waveform modes.
        The return value of this functions will be stored as
        self._sur_dimless()
        The __call__ function of self._sur_dimless() should take all inputs
        passed to self._sur_dimless() in the __call__ function of this class.
        """
        sur = new_surrogate.AlignedSpinCoOrbitalFrameSurrogateTidal()
        sur.load(self.h5filename)
        return sur

    def _get_intrinsic_parameters(self, q, chiA0, chiB0, precessing_opts,
            tidal_opts, par_dict):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, puts all intrinsic parameters of the surrogate
        into a single array.
        For example, for NRHybSur3dq8: x = [q, chiAz, chiBz].
        """
        if par_dict is not None:
            raise ValueError('Expected par_dict to be None.')
        Lambda1 = tidal_opts['Lambda1']
        Lambda2 = tidal_opts['Lambda2']
        if Lambda1 < 0 or Lambda1 > 10000:
            raise Exception('Lambda1=%.3f is outside the valid range ' \
                '[0,10000]'%Lambda1)
        if Lambda2 < 0 or Lambda2 > 10000:
            raise Exception('Lambda2=%.3f is outside the valid range ' \
                '[0,10000]'%Lambda2)

        x = [q, chiA0[2], chiB0[2], Lambda1, Lambda2]
        return x

class NRHybSur3dq8_CCE(NRHybSur3dq8):
    """
A class for the NRHybSur3dq8_CCE surrogate model presented in arXiv:2306.03148.

Evaluates gravitational waveforms generated by aligned-spin binary black hole
systems. This model was built using CCE numerical relativity (NR) waveforms that
have been hybridized using post-Newtonian (PN) and effective one body (EOB)
waveforms.

Unlike NRHybSur3dq8, NRHybSur3dq8_CCE captures memory effects.

This model includes the following spin-weighted spherical harmonic modes:
(2,2), (2,1), (2,0), (3,3), (3,2), (3,0), (4,4) (4,3), (4,0), and (5,5).
The m<0 modes are deduced from the m>0 modes.

The parameter space of validity is:
q \in [1, 10] and chi1z/chi2z \in [-1, 1],
where q is the mass ratio and chi1z/chi2z are the spins of the heavier/lighter
BH, respectively, in the direction of orbital angular momentum.

The surrogate has been trained in the range
q \in [1, 8] and chi1z/chi2z \in [-0.8, 0.8], but produces reasonable waveforms
in the above range and has been tested against existing NR waveforms in that
range.

See the __call__ method on how to evaluate waveforms.
   """
    pass

class NRHybSur2dq15(SurrogateEvaluator):
    """
A class for the NRHybSur2dq15 surrogate model presented in
arxiv:2203.10109. 

Evaluates gravitational waveforms generated by aligned-spin binary black hole
systems. This model was built using numerical relativity (NR) waveforms that
have been hybridized using effective one body (EOB) waveforms.

This model includes the following spin-weighted spherical harmonic modes:
(2,2), (2,1), (3,3), (4,4) and (5,5).
The m<0 modes are deduced from the m>0 modes.

The parameter space of validity is:
q \in [1, 20], |chi1z| \in [-0.7, 0.7], chi2z = 0.
where q is the mass ratio and chi1z/chi2z are the spins of the
heavier/lighter BH, respectively, in in the direction of orbital angular momentum.

The surrogate has been trained in the range
q \in [1, 15] and chi1z \in [-0.5, 0.5] chi2z = 0, but produces reasonable
waveforms in the above range. 

See the __call__ method on how to evaluate waveforms.
    """

    def __init__(self, h5filename):
        self.h5filename = h5filename
        domain_type = 'Time'
        keywords = {
            'Precessing': False,
            'Hybridized': True,
            }
        # soft_lims -> raise warning when outside lims
        # hard_lim -> raise error when outside lims
        # Format is [qMax, chiMax].
        soft_param_lims = [15.01, 0.5]
        hard_param_lims = [20.1, 0.71]
        super(NRHybSur2dq15, self).__init__(self.__class__.__name__, \
            domain_type, keywords, soft_param_lims, hard_param_lims)

    def _load_dimless_surrogate(self):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, handles the loading of the dimensionless surrogate.
        This should return the loaded surrogate.
        The loaded surrogate should have a __call__ function that returns the
        dimensionless time/frequency array and dimensionless waveform modes.
        The return value of this functions will be stored as
        self._sur_dimless()
        The __call__ function of self._sur_dimless() should take all inputs
        passed to self._sur_dimless() in the __call__ function of this class.
        """
        sur = new_surrogate.AlignedSpinCoOrbitalFrameSurrogate()
        sur.load(self.h5filename)
        return sur

    def _get_intrinsic_parameters(self, q, chiA0, chiB0, precessing_opts,
            tidal_opts, par_dict):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, puts all intrinsic parameters of the surrogate
        into a single array.
        For example, for NRHybSur2dq15: x = [q, chiAz].
        """
        if par_dict is not None:
            raise ValueError('Expected par_dict to be None.')

        if not chiB0[2] == 0:
            raise Exception('NRHybSur2dq15 assumes zero spin on secondary')
        x = [q, chiA0[2]]
        return x

class NRSur7dq4(SurrogateEvaluator):
    """
A class for the NRSur7dq4 surrogate model presented in Varma et al. 2019,
arxiv1905.09300.

Evaluates gravitational waveforms generated by precessing binary black hole
systems with generic mass ratios and spins.

This model includes the following spin-weighted spherical harmonic modes:
2<=ell<=4, -ell<=m<=ell.

The parameter space of validity is:
q \in [1, 6], and |chi1|,|chi2| \in [-1, 1], with generic directions.
where q is the mass ratio and chi1/chi2 are the spin vectors of the
heavier/lighter BH, respectively.

The surrogate has been trained in the range
q \in [1, 4] and |chi1|/|chi2| \in [-0.8, 0.8], but produces reasonable
waveforms in the above range and has been tested against existing
NR waveforms in that range.

See the __call__ method on how to evaluate waveforms.
    """

    def __init__(self, h5filename):
        self.h5filename = h5filename
        domain_type = 'Time'
        keywords = {
            'Precessing': True,
            }
        # soft_lims -> raise warning when outside lims
        # hard_lim -> raise error when outside lims
        # Format is [qMax, chiMax].
        soft_param_lims = [4.01, 0.801]
        hard_param_lims = [6.01, 1]
        super(NRSur7dq4, self).__init__(self.__class__.__name__, \
            domain_type, keywords, soft_param_lims, hard_param_lims)

    def _load_dimless_surrogate(self):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, handles the loading of the dimensionless surrogate.
        This should return the loaded surrogate.
        The loaded surrogate should have a __call__ function that returns the
        dimensionless time/frequency array and dimensionless waveform modes.
        The return value of this functions will be stored as
        self._sur_dimless()
        The __call__ function of self._sur_dimless() should take all inputs
        passed to self._sur_dimless() in the __call__ function of this class.
        See NRHybSur3dq8 for an example.
        """

        # needed to convert user input x to parameters used by surrogate fits
        def get_fit_params(x):
            """ Converts from x=[q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z]
                to x = [np.log(q), chi1x, chi1y, chiHat, chi2x, chi2y, chi_a]
                chiHat is defined in Eq.(3) of 1508.07253.
                chi_a = (chi1 - chi2)/2.
                Both chiHat and chi_a always lie in range [-1, 1].
            """

            x = np.copy(x)

            q = float(x[0])
            chi1z = float(x[3])
            chi2z = float(x[6])
            eta = q/(1.+q)**2
            chi_wtAvg = (q*chi1z+chi2z)/(1+q)
            chiHat = (chi_wtAvg - 38.*eta/113.*(chi1z + chi2z)) \
                /(1. - 76.*eta/113.)
            chi_a = (chi1z - chi2z)/2.

            x[0] = np.log(q)
            x[3] = chiHat
            x[6] = chi_a

            return x
        
        # needed to evaluate model-specific surrogate fits
        def get_fit_settings():
            """
            These are to rescale the mass ratio fit range
            from [-0.01, np.log(4+0.01)] to [-1, 1]. The chi fits are already
            in this range.


            Values defined here are model-specific. These are for NRSur7dq4.
            """

            q_fit_offset = -0.9857019407834238
            q_fit_slope = 1.4298059216576398
            q_max_bfOrder = 3
            chi_max_bfOrder = 2
            return q_fit_offset, q_fit_slope, q_max_bfOrder, chi_max_bfOrder

        # largest ell mode in the surrogate
        ellMax_NRSur7dq4 = 4

        # max allowable reference dimensionless orbital angular frequency
        omega_ref_max_NRSur7dq4 = 0.201

        sur = precessing_surrogate.PrecessingSurrogate(self.h5filename,
                 get_fit_params,get_fit_settings,ellMax_NRSur7dq4,omega_ref_max_NRSur7dq4)
        return sur

    def _get_intrinsic_parameters(self, q, chiA0, chiB0, precessing_opts,
            tidal_opts, par_dict):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, puts all intrinsic parameters of the surrogate
        into a single array.
        For example, for NRSur7dq4: x = [q, chiA0, chiB0].
        """
        x = [q, chiA0, chiB0]
        return x

class SEOBNRv4PHMSur(SurrogateEvaluator):
    """
A class for the SEOBNRv4PHM surrogate models.

Evaluates gravitational waveforms generated by precessing binary black hole
systems with generic mass ratios and spins.

This model includes the following spin-weighted spherical harmonic modes:
2<=ell<=5, -ell<=m<=ell in inertial frame.

The parameter space of validity is:
q \in [1, 20], and |chi1|,|chi2| \in [-0.99, 0.99], with generic directions.
where q is the mass ratio and chi1/chi2 are the spin vectors of the
heavier/lighter BH, respectively.

The surrogate has been trained in the range
q \in [1, 20] and |chi1|/|chi2| \in [-0.8, 0.8], but produces reasonable
waveforms in the above range and has been tested against existing
NR waveforms in that range.

See the __call__ method on how to evaluate waveforms.
In the __call__ method, x must have format x = [q, chi1, chi2].


IMPORTANT NOTES:
===============

The original SEOBNRv4PHM model (arXiv:2004.09442) parameterizes the 
direction of the BH spins relative to the Newtonian orbital angular momentum
vector of the binary in the co-precessing frame. The SEOBNRv4PHMSur model, 
however, parameterizes the spins relative to the direction where the radiation
is always strongest along the z-axis, and the (ell,m)= (2, ±2) modes are dominant.
This is the same convention used in the NRSur7dq4 surrogate model.

These two frames are different for non-precessing systems. Hence, specifying
the spins in the SEOBNRv4PHM-frame (e.g. when called through LALSimulation)
will not give a physically equivalent waveform when these same spin directions
are used in the SEOBNRv4PHMSur surrogate model. Ref. arXiv:2203.00381 provides
further discussion on this point.
    """

    def __init__(self, h5filename):
        self.h5filename = h5filename
        domain_type = 'Time'
        keywords = {
            'Precessing': True,
            }
        # soft_lims -> raise warning when outside lims
        # hard_lim -> raise error when outside lims
        # Format is [qMax, chiMax].
        soft_param_lims = [20.01, 0.801]
        hard_param_lims = [20.01, 0.99]
        super(SEOBNRv4PHMSur, self).__init__(self.__class__.__name__, \
            domain_type, keywords, soft_param_lims, hard_param_lims)

    def _load_dimless_surrogate(self):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, handles the loading of the dimensionless surrogate.
        This should return the loaded surrogate.
        The loaded surrogate should have a __call__ function that returns the
        dimensionless time/frequency array and dimensionless waveform modes.
        The return value of this functions will be stored as
        self._sur_dimless()
        The __call__ function of self._sur_dimless() should take all inputs
        passed to self._sur_dimless() in the __call__ function of this class.
        See NRHybSur3dq8 for an example.
        """

        def get_fit_params(x):
            """ convert user input x to parameters used by surrogate fits.
            
            Converts from x=[q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z]
                      to  x=[q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z]
            """
            ## Literally do nothing!!
            x = np.copy(x)
            return x
        
        def get_fit_settings(subdomain_ID):
            """ Get fit settings for SUBDOMAIN_ID.
            
            Arguments:
                subdomain_ID: The subdomain ID. Each subdomain has a uniquely
                            built surrogate model.
            Returns:
                q_fit_offset: The offset for the mass ratio (q) fit.
                q_fit_slope: The slope for the mass ratio (q) fit.
                q_max_bfOrder: The maximum order of the basis functions for fits in q.
                chi_max_bfOrder: The maximum order of the basis functions for fits in chi.
            """

            q_fit_offsets = [-2.5590551181102366,
                            -2.3809523809523814,
                            -3.125,
                            -3.125,
                            -3.125,
                            -4.829545454545455,
                            -4.829545454545455,
                            -4.829545454545455,
                            -4.528985507246377,
                            -4.528985507246377,
                            -4.528985507246377,
                            -6.340579710144926,
                            -6.340579710144926,
                            -6.340579710144926]
            q_fit_slopes = [1.5748031496062995,
                            0.7936507936507938,
                            0.5681818181818182,
                            0.5681818181818182,
                            0.5681818181818182,
                            0.5681818181818182,
                            0.5681818181818182,
                            0.5681818181818182,
                            0.3623188405797102,
                            0.3623188405797102,
                            0.3623188405797102,
                            0.36231884057971003,
                            0.36231884057971003,
                            0.36231884057971003]

            q_fit_offset    = q_fit_offsets[subdomain_ID]
            q_fit_slope     = q_fit_slopes[subdomain_ID]
            q_max_bfOrder   = 3
            chi_max_bfOrder = 2
            return q_fit_offset, q_fit_slope, q_max_bfOrder, chi_max_bfOrder

        def get_subdomain_ID(q, chiA0, chiB0):
            """ Get the surrogate subdomain ID for evaluation point in
             parameter space. 
             
            INPUT
            =====
            q :         Mass ratio, mA/mB >= 1.
            chiA0:      Dimensionless spin vector of the heavier black hole.
            chiB0:      Dimensionless spin vector of the lighter black hole.
            """
            chi1z = chiA0[2]
            chi2z = chiB0[2]
            chi_eff = (q*chi1z + chi2z) / (q + 1.)
            # print("q, chi_eff: ", q, chi_eff)
            if q <= 2.:
                return 0
            if q <= 4.:
                return 1
            if q <= 7. and chi_eff <= -0.3:
                return 2
            if q <= 7. and chi_eff <= 0.3:
                return 3
            if q <= 7. and chi_eff > 0.3:
                return 4
            if q <= 10. and chi_eff <= -0.3:
                return 5
            if q <= 10. and chi_eff <= 0.3:
                return 6
            if q <= 10. and chi_eff > 0.3:
                return 7
            if q <= 15. and chi_eff <= -0.3:
                return 8
            if q <= 15. and chi_eff <= 0.3:
                return 9
            if q <= 15. and chi_eff > 0.3:
                return 10
            if q > 15. and chi_eff <= -0.3:
                return 11
            if q > 15. and chi_eff <= 0.3:
                return 12
            if q > 15. and chi_eff > 0.3:
                return 13
            
            raise ValueError("Domain not found for q={}, chi1z={}, chi2z={}".format(q, chi1z, chi2z))


        # largest ell mode in the surrogate
        ellMax_SEOBNRv4PHMSur = 5

        # max allowable reference dimensionless orbital angular frequency
        omega_ref_max_SEOBNRv4PHMSur = 0.21

        # number of subdomains (should be maximum possible return of get_subdomain_ID + 1)
        num_subdomains_SEOBNRv4PHMSur = 14

        sur = precessing_surrogate.PrecessingSurrogateMultiDomain(self.h5filename,
              get_fit_params, get_fit_settings, ellMax_SEOBNRv4PHMSur,
              omega_ref_max_SEOBNRv4PHMSur, get_subdomain_ID,
              num_subdomains_SEOBNRv4PHMSur)
        return sur

    def _get_intrinsic_parameters(self, q, chiA0, chiB0, precessing_opts,
            tidal_opts, par_dict):
        """
        This function, which must be overriden for each derived class of
        SurrogateEvaluator, puts all intrinsic parameters of the surrogate
        into a single array.
        For example, for NRSur7dq4: x = [q, chiA0, chiB0].
        """
        x = [q, chiA0, chiB0]
        return x


#### for each model in the catalog (name or h5 file), associate class to load
#### NOTE: other classes maybe usable too, these just constitute
####       the default cases suitable for most people
SURROGATE_CLASSES = {
    "NRHybSur3dq8": NRHybSur3dq8,
    "NRHybSur3dq8_CCE": NRHybSur3dq8_CCE,
    "NRHybSur2dq15": NRHybSur2dq15,
    "NRSur7dq4": NRSur7dq4,
    "NRHybSur3dq8Tidal": NRHybSur3dq8Tidal,
    "SEOBNRv4PHMSur": SEOBNRv4PHMSur,
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
    def __new__(self, surrogate_name, surrogate_name_spliced=None):
        """ Returns a SurrogateEvaluator derived object based on name.

        INPUT
        =====
        SURROGATE_NAME: A string with either a surrogate's name (one of the
                        keys in SURROGATE_CLASSES dictionary) or the absolute
                        path to the surrogate's hdf5 file.

                        If h5 file is given, the surrogate's name is inferred
                        from the path.

                        If the surrogate's name is directly given, the
                        default surrogate download path is used to grab the
                        hdf5 file.

        SURROGATE_NAME_SPLICED: Certain models, like NRHybSur3dq8Tidal, modify
                                (or splice) an underlying model, in this case
                                NRHybSur3dq8. The same hdf5 file is used for both
                                models, which means one cannot directly load
                                the NRHybSur3dq8Tidal model from an hdf5 file
                                path.

                                If you wish to load a spliced model from its h5
                                file, provide (i) the hdf5 file path as its
                                surrogate name and (ii) the model name (e.g.
                                NRHybSur3dq8Tidal) as SURROGATE_NAME_SPLICED."""


        # the "output" of this if-block is surrogate_h5file and surrogate_name
        # to be used for "SURROGATE_CLASSES[surrogate_name](surrogate_h5file)"
        if surrogate_name.endswith('.h5'):
            # If h5 file is given, use that directly. But get the
            # surrogate_name used to pick from SURROGATE_CLASSES from the
            # filename
            surrogate_h5file = surrogate_name
            surrogate_name = os.path.basename(surrogate_h5file)
            surrogate_name = surrogate_name.split('.h5')[0]


            # check that value of SURROGATE_NAME_SPLICED is valid
            if surrogate_name_spliced is not None:
              assert(surrogate_name_spliced in ["NRHybSur3dq8Tidal"])
              surrogate_name = surrogate_name_spliced
        else:
            # If not, look for surrogate data in surrogate download_path

            if (surrogate_name=="NRHybSur3dq8Tidal"):
                # Special case for tidal model since it uses a NRHybSur3dq8 as
                # the base for the BBH part of the waveform
                surrogate_h5file = '%s/NRHybSur3dq8.h5'%(catalog.download_path())
                if not os.path.isfile(surrogate_h5file):
                    raise Exception("Surrogate data not found. Do"
                        " gwsurrogate.catalog.pull(NRHybSur3dq8)")
                #return NRHybSur3dq8Tidal(surrogate_h5file)
            elif (surrogate_name=='SEOBNRv4PHMSur'):
                surrogate_h5file = '%s/SEOBNRv4PHMSur.h5'%(catalog.download_path())
            else:
                surrogate_h5file = '%s/%s.h5'%(catalog.download_path(), \
                    surrogate_name)
                if not os.path.isfile(surrogate_h5file):
                    print("Surrogate data not found for %s. Downloading now."%surrogate_name)
                    catalog.pull(surrogate_name)

        if surrogate_name not in SURROGATE_CLASSES.keys():
            raise Exception('Invalid surrogate : %s'%surrogate_name)
        else:
            return SURROGATE_CLASSES[surrogate_name](surrogate_h5file)
