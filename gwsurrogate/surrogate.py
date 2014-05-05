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
from pylab import matplotlib as plt
import time


##############################################
class File:
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path, mode='r'):
		self.path = path
		self.mode_options = ['r', 'w', 'w+', 'a']
		self.open(mode=mode)
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
	"""Evaluate single-mode surrogate in terms of the function's amplitude and phase"""

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path, mode='r'):
		self.path = path
		self.mode = mode
		File.__init__(self, path, mode=mode)
		
		if mode == 'r':
			try:
				surr_name = path.split('/')[-2]
				if self.SurrogateID() != surr_name:
					print "\n>>> Warning: SurrogateID does not have expected name.\n"
			except: 
				print "\n>>> Warning: No SurrogateID found!\n"
		
		self.switch_dict = {
				'SurrogateID': self.SurrogateID,
				'tmin': self.tmin,
				'tmax': self.tmax,
				'dt': self.dt,
				'B': self.B,
				'eim_indices': self.eim_indices,
				'greedy_points': self.greedy_points,
				'V': self.V,
				'R': self.R,
				'fit_min': self.fit_min,
				'fit_max': self.fit_max,
				'affine_map': self.affine_map,
				'fitparams_amp': self.fitparams_amp,
				'fitparams_phase': self.fitparams_phase,
			}
		self.options = self.switch_dict.keys()
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __call__(self, option):
		return self.switch_dict[option]()
	
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

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def SurrogateID(self):
		if self.isopen():
			id = self.file['SurrogateID'][()]
			return ''.join(chr(cc) for cc in id)
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def tmin(self):
		if self.isopen():
			return self.file['tmin'][()]
		pass

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def tmax(self):
		if self.isopen():
			return self.file['tmax'][()]
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def dt(self):
		if self.isopen():
			return self.file['dt'][()]
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def B(self):
		if self.isopen():
			return self.file['B'][:]
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def eim_indices(self):
		if self.isopen():
			return self.file['eim_indices'][:]
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def greedy_points(self):
		if self.isopen():
			return self.file['greedy_points'][:]
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def V(self):
		if self.isopen():
			return self.file['V'][:]
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def R(self):
		if self.isopen():
			return self.file['R'][:]
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def fit_min(self):
		if self.isopen():
			return self.file['fit_min'][()]
		pass

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def fit_max(self):
		if self.isopen():
			return self.file['fit_max'][()]
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def affine_map(self):
		if self.isopen():
			return self.file['affine_map'][()]
		pass

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def fitparams_amp(self):
		if self.isopen():
			return self.file['fitparams_amp'][:]
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def fitparams_phase(self):
		if self.isopen():
			return self.file['fitparams_phase'][:]
		pass


##############################################
class TextSurrogate:
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, sdir):
		"""initialize single-mode surrogate from text files located in directory sdir"""

		### Surrogate directory ####
		self.SurrogateID = sdir

		### Surrogate's sampling rate and mass ratio (for fits) ###
		time_info             = np.loadtxt(sdir+'time_info.txt')
		self.q_interval       = np.loadtxt(sdir+'q_fit.txt')
		#self.Mtot             = np.loadtxt(sdir+'Mtot.txt')

		### unpack time info ###
		self.dt      = time_info[2]
		self.tmin    = time_info[0]
		self.tmax    = time_info[1]
		self.t_units = 'TOverMtot' # TODO: pass this through time_info for flexibility

		### greedy points (ordered by RB selection) ###
		self.greedypts = np.loadtxt(sdir+'greedy_points.txt')

		### empirical time index (ordered by EIM selection) ###
		self.eim_indx = np.loadtxt(sdir+'eim_indices.txt')

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

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __call__(self,q_eval):
		"""evaluate surrogate at q_eval"""

		### Allocate memory for polynomial fit evaluations ###
		Amp_eval   = np.zeros([self.dim_rb,1])
		phase_eval = np.zeros([self.dim_rb,1])

		### Map q_eval to the standard interval ?? ###
		if(self.affine_map):
			qmin = self.q_interval[0]
			qmax = self.q_interval[1]
			q_0 = 2*(q_eval - qmin)/(qmax - qmin) - 1;
		else:
			q_0 = q_eval

		### Evaluate fits ###
		for jj in range(self.dim_rb):
			Amp_eval[jj]     = np.polyval(self.AmpCoeff[jj,0:self.dim_rb],q_0)
			phase_eval[jj]   = np.polyval(self.PhaseCoeff[jj,0:self.dim_rb],q_0)

		### Build dim_RB-vector fit evalution of h ###
		h_EIM = Amp_eval*np.exp(-1j*phase_eval)

		### Surrogate modes hp and hc ###
		surrogate = np.dot(self.B,h_EIM)
		hp = surrogate.real
		hp = hp.reshape([self.time_samples,])
		hc = surrogate.imag
		hc = hc.reshape([self.time_samples,])

		times = np.arange(self.time_samples) * self.dt

		return times, hp, hc

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def plot(self,q_eval,timeM=False):
		"""plot surrogate evaluated at q_eval"""

		times, hp, hc = self(q_eval)
		if(self.t_units is 'TOverMtot'):
			#times = self.solarmass_over_mtot(times)
			xlab = '$t/M$'
		else:
			xlab = '$t$ (sec)'

		plt.pyplot.plot(times,hp)
		plt.pyplot.hold
		plt.pyplot.plot(times,hc)
		plt.pyplot.legend(['h plus', 'h cross'])
		plt.pyplot.xlabel(xlab)
		plt.pyplot.show()
		pass

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def h_rb(self,i):
		"""generate the ith reduced basis waveform"""
		times, hp, hc = self(self.greedypts[i])
		return times, hp, hc

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def plot_rb(self,i):
		"""plot the ith reduced basis waveform"""
		self.plot(self.greedypts[i])
		pass

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def timer(self,N):
		"""average time to evaluate n waveforms"""

		qmin = self.q_interval[0]
		qmax = self.q_interval[1]
		ran = np.random.uniform(qmin,qmax,N)
		q_ran = 2*(ran - qmin)/(qmax - qmin) - 1;

		tic = time.time()
		for i in ran:
			t,hp,hc = self(i)

		toc = time.time()
		print 'total time = ',toc-tic
		print 'average tme = ', (toc-tic)/float(N)
		pass

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def writetxt(self,q_eval,filename='output'):
		"""write waveform to text file"""
		times, hp, hc = self(q_eval)
		np.savetxt(filename,[times, hp, hc])
		pass

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def writebin(self,q_eval,filename='output.txt'):
		"""write waveform to numpy binary file"""
		times, hp, hc = self(q_eval)
		np.save(filename,[times, hp, hc])
		pass



	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def switch_time_to_secs(self,Mtot):
		"""switch from time units 'TOverMtot' to 's' """
		# TODO: temporal untis below should be implimented in a better way

		if(self.t_units is 'secs'):
			print 'times already in seconds'
		else:
			self.t_units = 'secs'
			self.solarmass_over_mtot_to_s(Mtot)
		pass
		
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def switch_time_to_TOverM(self,Mtot):
		"""switch from time units 's' to 'TOverMtot'"""

		if(self.t_units is 'TOverMtot'):
			print 'times alrady in t/M'
		else:
			self.t_units = 'TOverMtot'
			self.s_to_solarmass_over_mtot(Mtot)
		pass

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def s_to_solarmass_over_mtot(self,Mtot):
		"""convert seconds t to dimensionless t/Mtot"""
		self.dt   = (self.dt / mks.Msuninsec) / Mtot
		self.tmin = (self.tmin / mks.Msuninsec) / Mtot
		self.tmax = (self.tmax / mks.Msuninsec) / Mtot
		pass

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def solarmass_over_mtot_to_s(self,Mtot):
		"""convert dimensionless t/Mtot to seconds"""
		self.dt   = (self.dt * mks.Msuninsec) * Mtot
		self.tmin = (self.tmin * mks.Msuninsec) * Mtot
		self.tmax = (self.tmax * mks.Msuninsec) * Mtot
		pass
		

##############################################
class EvaluateSurrogate(HDF5Surrogate, TextSurrogate):

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path):
		ext = path.split('.')[-1]
		if ext == 'hdf5' or ext == 'h5':
			HDF5Surrogate.__init__(self, path)
		else:
			TextSurrogate.__init__(self, path)
			pass
		
		self.SurrogateID = self.SurrogateID()
		self.B = self.B()
		self.eim_indices = self.eim_indices()
		self.greedy_points = self.greedy_points()
		self.dt = self.dt()
		self.tmin = self.tmin()
		self.tmax = self.tmax()
		self.affine_map = self.affine_map()
		self.fitparams_amp = self.fitparams_amp()
		self.fitparams_phase = self.fitparams_phase()
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def time(self):
		return np.arange(self.tmin, self.tmax+self.dt, self.dt)

