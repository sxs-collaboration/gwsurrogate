""" Gravitational Wave Surrogate classes for text and hdf5 files"""

from __future__ import division

__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@umd.edu, crgalley@tapir.caltech.edu"
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
from scipy.interpolate import splrep
from scipy.interpolate import splev
import const_mks as mks
import matplotlib.pyplot as plt
import time
import os as os

try:
	import h5py
	h5py_enabled = True
except ImportError:
	h5py_enabled = False


##############################################
class File:
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path, mode='r'):
		self.path = path
		self.mode_options = ['r', 'w', 'w+', 'a']
		
		# Get all keys (e.g., variable names) in HDF5 file
		if mode == 'r':
			self.open(self.path, mode=mode)
			self.keys = self.file.keys()
		
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def open(self, path, mode='r'):
		if mode in self.mode_options:
			try: 
				self.file = h5py.File(path, mode)
				self.flag = 1
			except IOError:
				print "Could not open file."
				self.flag = 0
		else:
			raise Exception, "File action not recognized."
		
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def isopen(self):
		if self.flag == 1:
			return True
		else:
			return False
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def close(self):
		self.file.close()
		self.flag = 0
		pass


##############################################
class HDF5Surrogate(File):
	"""Load or export a single-mode surrogate in terms of the function's amplitude and phase from HDF5 data format"""

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path, mode='r'):
		self.path = path
		self.mode = mode
		File.__init__(self, path, mode=mode)
		
		if mode == 'r':
			
			### Get SurrogateID ####
			try:
				surr_name = path.split('/')[-2]
				if self.SurrogateID() != surr_name:
					print "\n>>> Warning: SurrogateID does not have expected name.\n"
			except: 
				print "\n>>> Warning: No SurrogateID found!\n"
		
			if self.isopen():
				
				### Unpack time info ###
				self.tmin = self.file['tmin'][()]
				self.tmax = self.file['tmax'][()]
				self.dt = self.file['dt'][()]
				
				if 't_units' in self.keys:
					self.t_units = self.file['t_units'][()]
				else:
					self.t_units = 'TOverMtot'
				
				### Greedy points (ordered by RB selection) ###
				self.greedy_points = self.file['greedy_points'][:]
				
				### Empirical time index (ordered by EIM selection) ###
				self.eim_indices = self.file['eim_indices'][:]
				
				### Complex B coefficients ###
				self.B = self.file['B'][:]	
				
				### Information about phase/amp parametric fit ###
				self.affine_map = self.file['affine_map'][()]
				self.fitparams_amp = self.file['fitparams_amp'][:]
				self.fitparams_phase = self.file['fitparams_phase'][:]
				self.fit_min = self.file['fit_min'][()]
				self.fit_max = self.file['fit_max'][()]
				self.fit_interval = [self.fit_min, self.fit_max]
			
				### Vandermonde V such that E (orthogonal basis) is E = BV ###
				self.V = self.file['V'][:]
				
				### R matrix such that waveform basis H = ER ###
				self.R = self.file['R'][:]
				
				### Transpose matrices if surrogate was built using ROMpy ###
				Bshape = np.shape(self.B)
				if Bshape[0] < Bshape[1]:
					self.B = np.transpose(self.B)
					self.V = np.transpose(self.V)
					self.R = np.transpose(self.R)
				
				### Deduce sizes from B ###
				self.time_samples = Bshape[0]
				self.dim_rb       = Bshape[1]
				
			else:
				raise Exception, "File not in write mode or is closed."		
			
			self.close()
			
		pass
		
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def SurrogateID(self):
		if self.isopen():
			id = self.file['SurrogateID'][()]
			return ''.join(chr(cc) for cc in id)
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def write_h5(self, id, t, B, eim_indices, greedy_points, fit_min, fit_max, affine_map, \
				fitparams_amp, fitparams_phase, V, R):
		""" Write surrogate data in standard format.
		
		Input:
		======
			surrogate_id    -- SurrogateID which should match the parent directory name 
			t               -- time series array (only min, max and increment saved)
			B               -- empirical interpolant operator (`B matrix`)
			eim_indices     -- indices of empirical nodes from time series array `t`
			greedy_points   -- parameters selected by reduced basis greedy algorithm
			V               -- Generalized Vandermonde matrix from empirical 
			                   interpolation method
			R               -- matrix coefficients relating the reduced basis to the 
			                   selected waveforms
			fit_min         -- min values of parameters used for surrogate fitting
			fit_max         -- max values of parameters used for surrogate fitting
			affine_map      -- mapped parameter domain to reference interval for fitting? 
			                   (True/False)
			fitparams_amp   -- fitting parameters for waveform amplitude
			fitparams_phase -- fitting parameters for waveform phase
		"""
		
		# Open file for writing. Filename based on surrogate ID.
		self.open(self.path+str(id)+'.h5', mode='w')
		
		# Write surrogate data to file
		surrogate_id = [ord(cc) for cc in id]
		self.file.create_dataset('SurrogateID', data=surrogate_id, dtype='int')
		
		self.file.create_dataset('tmin', data=t.min(), dtype='double')
		self.file.create_dataset('tmax', data=t.max(), dtype='double')
		self.file.create_dataset('dt', data=t[1]-t[0], dtype='double')
		
		self.file.create_dataset('B', data=B, dtype=B.dtype, compression='gzip')
		self.file.create_dataset('eim_indices', data=eim_indices, dtype='int', compression='gzip')
		self.file.create_dataset('greedy_points', data=greedy_points, dtype='double', compression='gzip')
		self.file.create_dataset('V', data=V, dtype=V.dtype, compression='gzip')
		self.file.create_dataset('R', data=R, dtype=R.dtype, compression='gzip')
		
		self.file.create_dataset('fit_min', data=fit_min, dtype='double')
		self.file.create_dataset('fit_max', data=fit_max, dtype='double')
		self.file.create_dataset('affine_map', data=affine_map, dtype='bool')
		self.file.create_dataset('fitparams_amp', data=fitparams_amp, dtype='double', compression='gzip')
		self.file.create_dataset('fitparams_phase', data=fitparams_phase, dtype='double', compression='gzip')
		
		self.close()
		
		pass


##############################################
class TextSurrogate:
	"""Load or export a single-mode surrogate in terms of the function's amplitude and phase from TEXT format"""

	### Files which define a text-based surrogate (both read and write) ###

	#variable_name_file   = file_name
	_time_info_file       = 'time_info.txt'
	_fit_interval_file    = 'q_fit.txt'
	_greedy_points_file   = 'greedy_points.txt'
	_eim_indices_file     = 'eim_indices.txt'
	_B_i_file             = 'B_imag.txt'
	_B_r_file             = 'B_real.txt'
	_fitparams_phase_file = 'fit_coeff_phase.txt'
	_fitparams_amp_file   = 'fit_coeff_amp.txt'
	_affine_map_file      = 'affine_map.txt'
	_V_i_file             = 'V_imag.txt'
	_V_r_file             = 'V_real.txt'
	_R_i_file             = 'R_im.txt'
	_R_r_file             = 'R_re.txt'
	_fitparams_norm_file  = 'fit_coeff_norm.txt'


	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, sdir):
		"""initialize single-mode surrogate defined by text files located in directory sdir"""

		### sdir is defined to the the surrogate's ID ###e
		self.SurrogateID = sdir


	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def load_text(self):

		sdir = self.SurrogateID

		### Surrogate's sampling rate and mass ratio (for fits) ###
		time_info         = np.loadtxt(sdir+self._time_info_file)
		self.fit_interval = np.loadtxt(sdir+self._fit_interval_file)
		#self.Mtot        = np.loadtxt(sdir+'Mtot.txt')

		### unpack time info ###
		self.dt      = time_info[2]
		self.tmin    = time_info[0]
		self.tmax    = time_info[1]
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

		### Vandermonde V such that E (orthogonal basis) is E = BV ###
		V_i    = np.loadtxt(sdir+self._V_i_file)
		V_r    = np.loadtxt(sdir+self._V_r_file)
		self.V = V_r + (1j)*V_i

		### R matrix such that waveform basis H = ER ###
		R_i    = np.loadtxt(sdir+self._R_i_file)
		R_r    = np.loadtxt(sdir+self._R_r_file)
		self.R = R_r + (1j)*R_i

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def np_savetxt_safe(self,fname,data):
		""" numpys savetext without overwrites """

		if os.path.isfile(fname):
			raise Exception, "file already exists"
		else: 
			np.savetxt(fname,data)

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def write_text(self, t, B, eim_indices, greedy_points, fit_min, fit_max, affine_map, \
				fitparams_amp, fitparams_phase, fitparams_norm, V, R):
		""" Write surrogate data (text) in standard format.
		
		Input:
		======
			t               -- time series array (only min, max and increment saved)
			B               -- empirical interpolant operator (`B matrix`)
			eim_indices     -- indices of empirical nodes from time series array `t`
			greedy_points   -- parameters selected by reduced basis greedy algorithm
			V               -- Generalized Vandermonde matrix from empirical 
			                   interpolation method
			R               -- matrix coefficients relating the reduced basis to the 
			                   selected waveforms
			fit_min         -- min values of parameters used for surrogate fitting
			fit_max         -- max values of parameters used for surrogate fitting
			affine_map      -- mapped parameter domain to reference interval for fitting? 
			                   (True/False)
			fitparams_amp   -- fitting parameters for waveform amplitude
			fitparams_phase -- fitting parameters for waveform phase
			fitparams_norm  -- fitting parameters for waveform norm
		"""

		# TODO: flag to zip folder with tar -cvzf SURROGATE_NAME.tar.gz SURROGATE_NAME/

		### pack mass ratio interval (for fits) and time info ###
		q_fit     = [fit_min, fit_max]
		dt        = t[3] - t[2]
		time_info = [t[0], t[-1], dt] # tmin, tmax, dt


		self.np_savetxt_safe(self.SurrogateID+self._fit_interval_file,q_fit)
		self.np_savetxt_safe(self.SurrogateID+self._time_info_file,time_info)
		self.np_savetxt_safe(self.SurrogateID+self._greedy_points_file,greedy_points)
		self.np_savetxt_safe(self.SurrogateID+self._eim_indices_file,eim_indices)
		self.np_savetxt_safe(self.SurrogateID+self._B_i_file,B.imag)
		self.np_savetxt_safe(self.SurrogateID+self._B_r_file,B.real)
		self.np_savetxt_safe(self.SurrogateID+self._fitparams_phase_file,fitparams_phase)
		self.np_savetxt_safe(self.SurrogateID+self._fitparams_amp_file,fitparams_amp)
		self.np_savetxt_safe(self.SurrogateID+self._affine_map_file,np.array([int(affine_map)]) )
		self.np_savetxt_safe(self.SurrogateID+self._V_i_file,V.imag)
		self.np_savetxt_safe(self.SurrogateID+self._V_r_file,V.real)
		self.np_savetxt_safe(self.SurrogateID+self._R_i_file,R.imag)
		self.np_savetxt_safe(self.SurrogateID+self._R_r_file,R.real)
		self.np_savetxt_safe(self.SurrogateID+self._fitparams_norm_file,fitparams_norm)

		pass


##############################################
class ExportSurrogate(HDF5Surrogate, TextSurrogate):
	"""Export single-mode surrogate"""
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path):
		
		# export HDF5 or Text surrogate data depending on input file extension
		ext = path.split('.')[-1]
		if ext == 'hdf5' or ext == 'h5':
			HDF5Surrogate.__init__(self, path)
		else:

			if( not(path[-1:] is '/') ):
                                raise Exception, "path name should end in /"

			try:
				os.mkdir(path)
				print "Successfully created a surrogate directory...use write_text to export your surrogate!"
				TextSurrogate.__init__(self, path)	

			except OSError:
				print "Could not create a surrogate directory. Not ready to export, please try again."


##############################################
class EvaluateSurrogate(File, HDF5Surrogate, TextSurrogate):
	"""Evaluate single-mode surrogate in terms of the function's amplitude and phase"""
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path, deg=3):
		
		# Load HDF5 or Text surrogate data depending on input file extension
		ext = path.split('.')[-1]
		if ext == 'hdf5' or ext == 'h5':
			HDF5Surrogate.__init__(self, path)
		else:
			TextSurrogate.__init__(self, path)
			self.load_text()

		# Time samples associated with the original data used to build the surrogate
		self.times = np.arange(self.tmin, self.tmax+self.dt, self.dt)
		
		# Interpolate columns of the empirical interpolant operator, B, using cubic spline
		self.reB_spline_params = [splrep(self.times, self.B[:,jj].real, k=deg) for jj in range(self.dim_rb)]
		self.imB_spline_params = [splrep(self.times, self.B[:,jj].imag, k=deg) for jj in range(self.dim_rb)]
		
		# Convenience for plotting purposes
		self.plt = plt
		
		pass

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __call__(self, q_eval, M_eval=None, dist_eval=None, phi_ref=None, f_low=None, samples=None):
		"""Return surrogate evaluation for...
			q    = mass ratio (dimensionless) 
			M    = total mass (solar masses) 
			dist = distance to binary system (megaparsecs)
			phir = mode's phase at peak amplitude
			flow = starting instantaneous frequency, will check if f_start < flow """


		### evaluate rh/M surrogate. Physical surrogates are generated by applying additional operations. ###
		hp, hc = self.h_sur(q_eval, samples=samples)


		### adjust phase if requested -- see routine for assumptions about mode's peak ###
		if (phi_ref is not None):
			h  = self.adjust_merger_phase(hp + 1.0j*hc,phi_ref)
			hp = h.real
			hc = h.imag


		### if (q,M,distance) requested, use scalings and norm fit to get a physical mode ###
		if( M_eval is not None and dist_eval is not None):
			amp0    = ((M_eval * mks.Msun ) / (dist_eval * mks.Mpcinm )) * ( mks.G / np.power(mks.c,2.0) )
			t_scale = mks.Msuninsec * M_eval
		else:
			amp0    = 1.0
			t_scale = 1.0

		hp     = amp0 * hp
		hc     = amp0 * hc
		t      = self.time()
		t      = t_scale * t


		### check that surrogate's starting frequency is below f_low, otherwise throw a warning ###
		if f_low is not None:
			self.find_instant_freq(hp, hc, t, f_low)

		return t, hp, hc

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def affine_mapper_checker(self, q_eval):
		"""map q_eval to the standard interval [-1,1] if necessary. Also check is q_eval within training interval"""

		qmin, qmax = self.fit_interval

		if( q_eval < qmin or q_eval > qmax):
			print "Warning: Surrogate not trained at requested parameter value" # needed to display in ipython notebook
			Warning("Surrogate not trained at requested parameter value")


		if self.affine_map:
			q_0 = 2.*(q_eval - qmin)/(qmax - qmin) - 1.;
		else:
			q_0 = q_eval

		return q_0


	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def norm_eval(self, q_0,affine_mapped=True):
		"""evaluate norm fit"""

		if( not(affine_mapped) ):
			q_0 = self.affine_mapper_checker(q_0)

		nrm_eval  = np.array([ np.polyval(self.fitparams_norm, q_0) ])
		return nrm_eval

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def h_sur(self, q_eval, samples=None):
		"""evaluate surrogate at q_eval. This returns dimensionless rh/M waveforms in units of t/M."""

		### Map q_eval to the standard interval and check parameter validity ###
		q_0 = self.affine_mapper_checker(q_eval)

		### Evaluate amp/phase/norm fits ###
		amp_eval   = np.array([ np.polyval(self.fitparams_amp[jj, 0:self.dim_rb], q_0) for jj in range(self.dim_rb) ])
		phase_eval = np.array([ np.polyval(self.fitparams_phase[jj, 0:self.dim_rb], q_0) for jj in range(self.dim_rb) ])
		nrm_eval   = self.norm_eval(q_0)

		### Build dim_RB-vector fit evaluation of h ###
		h_EIM = amp_eval*np.exp(1j*phase_eval)

		### Surrogate modes hp and hc ###
		if samples == None:
			surrogate = np.dot(self.B, h_EIM)
		else:
			surrogate = np.dot(self.resample_B(samples), h_EIM)

		surrogate = nrm_eval * surrogate
		hp = surrogate.real
		#hp = hp.reshape([self.time_samples,])
		hc = surrogate.imag
		#hc = hc.reshape([self.time_samples,])

		return hp, hc
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def find_instant_freq(self, hp, hc, t, f_low = None):
		"""instantaneous frequency at t_start for 

                   h = A(t) exp(2 * pi * i * f(t) * t), 

                   where \partial_t A ~ \partial_t f ~ 0. If f_low passed will check its been achieved."""

		h    = hp + 1j*hc
		dt   = t[1] - t[0]
		hdot = (h[2] - h[0]) / (2 * dt) # 2nd order derivative approximation at t[1]

		f_instant = hdot / (2 * np.pi * 1j * h[1])
		f_instant = f_instant.real

		if f_low is None:
			return f_instant
		else:

			if f_instant > f_low:
				raise Warning, "starting frequency is "+str(f_instant)
			else:
				pass


	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def amp_phase(self,h):
        	"""Get amplitude and phase of waveform, h = A*exp(i*phi)"""

		amp = np.abs(h);
		return amp, np.unwrap( np.real( -1j * np.log( h/amp ) ) )


	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def phi_merger(self,h):
        	"""Phase of mode (typically 22) at amplitude's peak. h = A*exp(i*phi). Routine assumes peak is exactly on temporal grid, which it is for rh/M surrogates."""

		amp, phase = self.amp_phase(h)
		argmax_amp = np.argmax(amp)

		return phase[argmax_amp]


	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def adjust_merger_phase(self,h,phiref):
		"""Modify GW mode's phase such that at time of amplitude peak, t_peak, we have phase(t_peak) = phiref"""

		phimerger = self.phi_merger(h)
		phiadj    = phimerger - phiref

		return ( h*np.exp(-1.0j *phiadj) )

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def timer(self):
		"""average time to evaluate waveforms"""

		qmin, qmax = self.fit_interval
		ran = np.random.uniform(qmin, qmax, 1000)

		tic = time.time()
		for i in ran:
			hp, hc = self.h_sur(i)

		toc = time.time()
		print 'Timing results (results quoted in seconds)...'
		print 'Total time to generate 1000 waveforms = ',toc-tic
		print 'Average time to generate a single waveform = ', (toc-tic)/1000.0
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# TODO: this routine might not serve a useful purpose -- think about it
	def time(self, units=None):
		# NOTE: Is Mtot the total mass of the surrogate or the total mass one wants to 
		# evaluate the time?
		"""Return time samples in specified units.
		
		Options for units:
		====================
		None		-- Time in geometric units, G=c=1 (DEFAULT)
		'solarmass' -- Time in units of solar masses
		'sec'		-- Time in units of seconds
		"""
		t = self.times
		if units == 'solarmass':
			t *= mks.Msuninsec
		elif units == 'sec':
			t *= mks.Msuninsec
		return t
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def basis(self, i, flavor='waveform'):
		"""compute the ith cardinal, orthogonal, or (exact, as opposed to 
                   its surrogate approximate) waveform basis."""

		if flavor == 'cardinal':
			basis = self.B[:,i]
		elif flavor == 'orthogonal':
			basis = np.dot(self.B,self.V)[:,i]
		elif flavor == 'waveform':
			E = np.dot(self.B,self.V)
			basis = np.dot(E,self.R)[:,i]
		else:
			raise ValueError("Not a valid basis type")

		return basis

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def resample_B(self, samples):
		"""resample the empirical interpolant operator, B, at the input samples"""
		return np.array([splev(samples, self.reB_spline_params[jj])  \
				+ 1j*splev(samples, self.imB_spline_params[jj]) for jj in range(self.dim_rb)]).T
		
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def plot_pretty(self, time, hp, hc, fignum=1, flavor='regular'):
		"""create a waveform figure with nice formatting and labels.
		returns figure method for saving, plotting, etc."""

		# Plot waveform
		fig = self.plt.figure(fignum)
		ax = fig.add_subplot(111)

		if flavor == 'regular':
			self.plt.plot(time, hp, 'k-', label='$h_+ (t)$')
			self.plt.plot(time, hc, 'k--', label='$h_\\times (t)$')
		elif flavor == 'semilogy':
			self.plt.semilogy(time, hp, 'k-', label='$h_+ (t)$')
			self.plt.semilogy(time, hc, 'k--', label='$h_\\times (t)$')
		else:
			raise ValueError("Not a valid plot type")

		self.plt.xlabel('Time, $t/M$')
		self.plt.ylabel('Waveform')
		self.plt.legend(loc='upper left')

		return fig
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def plot_rb(self, i, showQ=True):
		"""plot the ith reduced basis waveform"""
		
		# NOTE: Need to allow for different time units for plotting and labeling
		
		# Compute surrogate approximation of RB waveform
		basis = self.basis(i)
		hp    = basis.real
		hc    = basis.imag
		
		# Plot waveform
		fig = self.plot_pretty(self.times,hp,hc)
		
		if showQ:
			self.plt.show()
		
		# Return figure method to allow for saving plot with fig.savefig
		return fig
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def plot_sur(self, q_eval, timeM=False, showQ=True):
		"""plot surrogate evaluated at q_eval"""
		
		hp, hc = self.h_sur(q_eval)
		t      = self.time()

		if self.t_units == 'TOverMtot':
			#times = self.solarmass_over_mtot(times)
			xlab = 'Time, $t/M$'
		else:
			xlab = 'Time, $t$ (sec)'

		# Plot surrogate waveform
		fig = self.plot_pretty(t,hp,hc)
		self.plt.xlabel(xlab)
		self.plt.ylabel('Surrogate waveform')
		
		if showQ:
			self.plt.show()
		
		# Return figure method to allow for saving plot with fig.savefig
		return fig
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def writetxt(self, q_eval, filename='output'):
		"""write waveform to text file"""
		hp, hc = self(q_eval)
		np.savetxt(filename, [self.times, hp, hc])
		pass

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def writebin(self, q_eval, filename='output.txt'): # .txt for binary?
		"""write waveform to numpy binary file"""
		hp, hc = self(q_eval)
		np.save(filename, [self.times, hp, hc])
		pass


