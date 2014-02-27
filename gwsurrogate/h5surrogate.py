# --- h5surrogate.py ---


import numpy as np, h5py


##############################################
class File:
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path, mode='r'):
		self.mode_options = ['r', 'w', 'w+', 'a']
		self.open(path, mode=mode)
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def open(self, path, mode):
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
	"""Evaluate single-mode surrogate in terms of the function's amplitude and phase"""

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self, path):
		File.__init__(self, path, 'r')
		#self.surrogate_id = self.file['SurrogateID'][()]
		#surr_name = path.split('/')[-2]
		#if self.surrogate_id != surr_name:
		#	print "\n>>> Warning: SurrogateID does not have expected name.\n"
		
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def dt(self):
		if self.isopen():
			return self.file['dt'][()]
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
	# q may not be a parameter for the surrogate (e.g., m1 and m2)...
	# Maybe put this in the instantiation part since this is independent of the online
	# call, more or less?
	def qmin_fit(self):
		pass
	def qmax_fit(self):
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
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def vandermonde(self):
		if self.isopen():
			return self.file['vandermonde'][:]
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def R(self):
		if self.isopen():
			return self.file['R'][:]
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
	
	
	
	
	