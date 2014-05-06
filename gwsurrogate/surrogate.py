""" Gravitational Wave Surrogate classes for text and hdf5 files"""

from __future__ import division

__copyright__ = "Copyright (C) 2014 GW surrogate group"
__email__     = "sfield@umd.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Scott Field, Chad Galley"

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

import numpy as np, h5py
import const_mks as mks
import matplotlib.pyplot as plt
import time


##############################################
class File:
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path, mode='r'):
		self.path = path
		self.mode_options = ['r', 'w', 'w+', 'a']
		self.open(mode=mode)
		
		# Get all keys (e.g., variable names) in HDF5 file
		self.keys = self.file.keys()
		
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def open(self, mode='r'):
		if mode in self.mode_options:
			try: 
				self.file = h5py.File(self.path, mode)
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
	"""Evaluate single-mode surrogate in terms of the function's amplitude and phase from HDF5 data format"""

	# NOTE: Restructure __call__ to evaluate surrogate from data in memory.

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path, mode='r'):
		self.path = path
		self.mode = mode
		File.__init__(self, path, mode=mode)
		
		keys = self.file.keys()
		
		### Get SurrogateID ####
		if mode == 'r':
			try:
				surr_name = path.split('/')[-2]
				if self.SurrogateID() != surr_name:
					print "\n>>> Warning: SurrogateID does not have expected name.\n"
			except: 
				print "\n>>> Warning: No SurrogateID found!\n"
		
		if self.isopen():
			
			### Was surrogate built using ROMpy? ###
			if 'from_rompy' in self.keys:
				self.from_rompy = self.file['from_rompy'][()]
			else:
				self.from_rompy = True 
			
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
			if self.from_rompy:
				self.B = np.transpose(self.B)
				self.V = np.transpose(self.V)
				self.R = np.transpose(self.R)
			
			### Deduce sizes from B ###
			Bshape = np.shape(self.B)
			self.dim_rb       = Bshape[1]
			self.time_samples = Bshape[0]
			
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
		
		if self.isopen() and self.mode == 'w':
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
		else:
			raise Exception, "File not in write mode or is closed."
		pass

	
##############################################
class TextSurrogate:
	"""Evaluate single-mode surrogate in terms of the function's amplitude and phase from TEXT format"""
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, sdir):
		"""initialize single-mode surrogate from text files located in directory sdir"""

		### Surrogate directory ####
		self.SurrogateID = sdir

		### Surrogate's sampling rate and mass ratio (for fits) ###
		time_info             = np.loadtxt(sdir+'time_info.txt')
		self.fit_interval       = np.loadtxt(sdir+'q_fit.txt')
		#self.Mtot             = np.loadtxt(sdir+'Mtot.txt')

		### unpack time info ###
		self.dt      = time_info[2]
		self.tmin    = time_info[0]
		self.tmax    = time_info[1]
		self.t_units = 'TOverMtot' # TODO: pass this through time_info for flexibility

		### greedy points (ordered by RB selection) ###
		self.greedy_points = np.loadtxt(sdir+'greedy_points.txt')

		### empirical time index (ordered by EIM selection) ###
		self.eim_indices = np.loadtxt(sdir+'eim_indices.txt')

		### Complex B coefficients ###
		B_i    = np.loadtxt(sdir+'B_imag.txt')
		B_r    = np.loadtxt(sdir+'B_real.txt')
		self.B = B_r + (1j)*B_i

		### Deduce sizes from B ###
		self.dim_rb       = B_r.shape[1]
		self.time_samples = B_r.shape[0]

		### Information about phase/amp parametric fit ###
		self.PhaseCoeff = np.loadtxt(sdir+'fit_coeff_phase.txt')
		self.AmpCoeff   = np.loadtxt(sdir+'fit_coeff_amp.txt')
		self.affine_map = bool(np.loadtxt(sdir+'affine_map.txt'))

		### Vandermonde V such that E (orthogonal basis) is E = BV ###
		V_i    = np.loadtxt(sdir+'V_imag.txt')
		V_r    = np.loadtxt(sdir+'V_real.txt')
		self.V = V_r + (1j)*V_i

		### R matrix such that waveform basis H = ER ###
		R_i    = np.loadtxt(sdir+'R_im.txt')
		R_r    = np.loadtxt(sdir+'R_re.txt')
		self.R = R_r + (1j)*R_i
		
		pass

#	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#	def switch_time_to_secs(self,Mtot):
#		"""switch from time units 'TOverMtot' to 's' """
#		# TODO: temporal untis below should be implimented in a better way
#
#		if(self.t_units is 'secs'):
#			print 'times already in seconds'
#		else:
#			self.t_units = 'secs'
#			self.solarmass_over_mtot_to_s(Mtot)
#		pass
#		
#	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#	def switch_time_to_TOverM(self,Mtot):
#		"""switch from time units 's' to 'TOverMtot'"""
#
#		if(self.t_units is 'TOverMtot'):
#			print 'times alrady in t/M'
#		else:
#			self.t_units = 'TOverMtot'
#			self.s_to_solarmass_over_mtot(Mtot)
#		pass
#
#	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#	def s_to_solarmass_over_mtot(self,Mtot):
#		"""convert seconds t to dimensionless t/Mtot"""
#		self.dt   = (self.dt / mks.Msuninsec) / Mtot
#		self.tmin = (self.tmin / mks.Msuninsec) / Mtot
#		self.tmax = (self.tmax / mks.Msuninsec) / Mtot
#		pass
#
#	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#	def solarmass_over_mtot_to_s(self,Mtot):
#		"""convert dimensionless t/Mtot to seconds"""
#		self.dt   = (self.dt * mks.Msuninsec) * Mtot
#		self.tmin = (self.tmin * mks.Msuninsec) * Mtot
#		self.tmax = (self.tmax * mks.Msuninsec) * Mtot
#		pass
#		


##############################################
class EvaluateSurrogate(HDF5Surrogate, TextSurrogate):
	"""Evaluate single-mode surrogate in terms of the function's amplitude and phase"""
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path):
		
		# Load HDF5 or Text surrogate data depending on input file extension
		ext = path.split('.')[-1]
		if ext == 'hdf5' or ext == 'h5':
			HDF5Surrogate.__init__(self, path)
		else:
			TextSurrogate.__init__(self, path)	
		
		# Time samples
		self.times = np.arange(self.tmin, self.tmax+self.dt, self.dt)
		
		# Convenience for plotting purposes
		self.plt = plt
		
		pass

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __call__(self, q_eval):
		return self.h_sur(q_eval)
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def h_sur(self, q_eval):
		"""evaluate surrogate at q_eval"""

		### Map q_eval to the standard interval ?? ###
		if self.affine_map:
			qmin, qmax = self.fit_interval
			q_0 = 2*(q_eval - qmin)/(qmax - qmin) - 1;
		else:
			q_0 = q_eval

		### Evaluate fits ###
		amp_eval = np.array([ np.polyval(self.fitparams_amp[jj, 0:self.dim_rb], q_0) for jj in range(self.dim_rb) ])
		phase_eval = np.array([ np.polyval(self.fitparams_phase[jj, 0:self.dim_rb], q_0) for jj in range(self.dim_rb) ])

		### Build dim_RB-vector fit evalution of h ###
		h_EIM = amp_eval*np.exp(1j*phase_eval)

		### Surrogate modes hp and hc ###
		surrogate = np.dot(self.B, h_EIM)
		hp = surrogate.real
		hp = hp.reshape([self.time_samples,])
		hc = surrogate.imag
		hc = hc.reshape([self.time_samples,])

		return hp, hc
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def timer(self, N):
		"""average time to evaluate n waveforms"""

		qmin, qmax = self.fit_interval
		ran = np.random.uniform(qmin, qmax, N)
		q_ran = 2.*(ran - qmin)/(qmax - qmin) - 1.;

		tic = time.time()
		for i in ran:
			hp, hc = self(i)

		toc = time.time()
		print 'total time = ',toc-tic
		print 'average time = ', (toc-tic)/float(N)
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
	def h_rb(self, i):
		"""generate a surrogate approximation for the ith reduced basis waveform"""
		hp, hc = self(self.greedy_points[i])
		return hp, hc
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def plot_rb(self, i, showQ=True):
		"""plot the surrogate approximation for the ith reduced basis waveform"""
		
		# NOTE: Need to allow for different time units for plotting and labeling
		
		# Compute surrogate approximation of RB waveform
		hp, hc = self.h_rb(i)
		
		# Plot waveform
		fig = self.plt.figure(1)
		ax = fig.add_subplot(111)
		self.plt.plot(self.times, hp, 'k-', label='$h_+ (t)$')
		self.plt.plot(self.times, hc, 'k--', label='$h_\\times (t)$')
		self.plt.xlabel('Time, $t/M$')
		self.plt.ylabel('Waveform')
		self.plt.legend(loc='upper left')
		
		if showQ:
			self.plt.show()
		
		# Return figure method to allow for saving plot with fig.savefig
		return fig
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def plot_sur(self, q_eval, timeM=False, showQ=True):
		"""plot surrogate evaluated at q_eval"""
		
		hp, hc = self(q_eval)
		if(self.t_units is 'TOverMtot'):
			#times = self.solarmass_over_mtot(times)
			xlab = '$t/M$'
		else:
			xlab = '$t$ (sec)'

		# Plot surrogate waveform
		fig = self.plt.figure(1)
		ax = fig.add_subplot(111)
		self.plt.plot(self.times, hp, 'k-', label='$h_+ (t)$')
		self.plt.plot(self.times, hc, 'k--', label='$h_\\times (t)$')
		self.plt.xlabel(xlab)
		self.plt.ylabel('Surrogate waveform')
		self.plt.legend(loc='upper left')
		
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
