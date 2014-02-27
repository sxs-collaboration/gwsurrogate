# --- h5surrogate.py ---


import numpy as np, h5py


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
# Just writing this out to make sure the overall structure is ok/understandable/not screwy...
class EvaluateSurrogate(HDF5Surrogate):	# Include TextSurrogate

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path):
		ext = path.split('.')[-1]
		if ext == 'hdf5' or ext == 'h5':
			HDF5Surrogate.__init__(self, path)
		else:
			# TextSurrogate.__init__(self, path)
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
	
	
	
	
	