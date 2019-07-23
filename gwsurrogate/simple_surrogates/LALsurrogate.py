""" This code as intended to be used to match the LAL API for calling
waveforms. Its far from finished and unlikely to be useful for 
new surrogate models.

Keep around for now, but consider deleting (7/2019))."""


from __future__ import division # for python 2


from scipy import interpolate
import numpy, h5py
import numpy.linalg as la
from pylab import *
from .surrogate import SurrogateGW

class Rescaling(SurrogateGW):
    """Class for rescaling a surrogate waveform by the desired total mass. 
    Interpolates the matrix of basis functions, B, and writes interpolants to file"""
    
    def __init__(self, surrogate_dir):
        SurrogateGW.__init__(self, surrogate_dir)
        self.surrogate_dir = surrogate_dir
    
    def interp_fn(self, fn):
        """Interpolate a real function, possibly shifting the zero to be at the peak amplitude"""
        times = np.arange(self.time_samples) * self.deltaT
        return interpolate.splrep(times, fn)
    
    def save_interp(self, compressQ=True):
        """Generate interpolant and save to file B_interp.hdf5"""
        f = h5py.File(self.surrogate_dir+'B_interp.hdf5', 'w')
        for jj in range(self.dim_rb):
            # Interpolate each basis function
            re_t,re_c,re_k = self.interp_fn(self.B[:,jj].real)
            im_t,im_c,im_k = self.interp_fn(self.B[:,jj].imag)
            basis_group = f.create_group('basis_'+str(jj))
            
            # Write to disk, possibly compressing the interpolant data
            if compressQ:
                basis_group.create_dataset('knots_real', dtype='double', data=re_t, compression='gzip')
                basis_group.create_dataset('coeff_real', dtype='double', data=re_c, compression='gzip')
                basis_group.create_dataset('knots_imag', dtype='double', data=im_t, compression='gzip')
                basis_group.create_dataset('coeff_imag', dtype='double', data=im_c, compression='gzip')
            else:
                basis_group.create_dataset('knots_real', dtype='double', data=re_t)
                basis_group.create_dataset('coeff_real', dtype='double', data=re_c)
                basis_group.create_dataset('knots_imag', dtype='double', data=im_t)
                basis_group.create_dataset('coeff_imag', dtype='double', data=im_c)
            basis_group.create_dataset('degree_real',dtype='int',data=re_k)
            basis_group.create_dataset('degree_imag',dtype='int',data=im_k)
        f.create_dataset('dim_rb', dtype=int, data=self.dim_rb)
        f.create_dataset('time_samples', dtype=int, data=self.time_samples)
        f.create_dataset('Mtot', dtype=double, data=self.Mtot)
        f.create_dataset('sample_rate', dtype=double, data=1./self.deltaT)
        f.close()
        return 
        
    def fit_amps(self):
        """Fit unnormalized amplitudes of greedy EOB waveforms"""
        pass

class LALSurrogateEOB(SurrogateGW):
    """Class for generating a surrogate waveform for different total masses"""
    
    def __init__(self, surrogate_dir):
        
        ### items specified by YOU ###
        SurrogateGW.__init__(self, surrogate_dir)
        
        # load mass ratio interval for which surrogate is valid
        self.q_interval = np.loadtxt(surrogate_dir+'qSpace.txt')
        self.qmin = self.q_interval[0]
        self.qmax = self.q_interval[1]

        file = h5py.File(surrogate_dir+'B_interp.hdf5', 'r')
        
        # load fiducial sample rate used for making the surrogate
        samplerate = file['sample_rate'][()]
        self.deltaT = 1.0/samplerate

        # load some parameters
        self.dim_rb = file['dim_rb'][()]
        self.time_samples = file['time_samples'][()]
        self.Mtot = file['Mtot'][()]
        
        # allocate memory for the interpolant of B(t)
        self.re_t = np.zeros([self.dim_rb, self.time_samples+3+1], dtype='double')
        self.re_c = np.zeros([self.dim_rb, self.time_samples+3+1], dtype='double')
        self.re_k = np.zeros(self.dim_rb, dtype='int')
        self.im_t = np.zeros([self.dim_rb, self.time_samples+3+1], dtype='double')
        self.im_c = np.zeros([self.dim_rb, self.time_samples+3+1], dtype='double')
        self.im_k = np.zeros(self.dim_rb, dtype='int')
        
        # load interpolants of basis functions
        for jj in range(self.dim_rb):
            self.re_t[jj] = file['basis_'+str(jj)+'/knots_real'][:]
            self.re_c[jj] = file['basis_'+str(jj)+'/coeff_real'][:]
            self.re_k[jj] = file['basis_'+str(jj)+'/degree_real'][()]
            self.im_t[jj] = file['basis_'+str(jj)+'/knots_imag'][:]
            self.im_c[jj] = file['basis_'+str(jj)+'/coeff_imag'][:]
            self.im_k[jj] = file['basis_'+str(jj)+'/degree_imag'][()]
        file.close()
        
        # load coefficients of polynomial parametric fit for phase
        self.PhaseCoeff = np.loadtxt(surrogate_dir+'PhasePolyCoeff.txt')

        # load coefficients of polynomial parametric fit for amplitude
        self.AmpCoeff = np.loadtxt(surrogate_dir+'AmpPolyCoeff.txt')
    
    def argpeak(self, h):
        """Get array index of a waveform's peak amplitude"""
        return np.argmax(np.abs(h))
    
    def shift(self, h, di):
        """Shift waveform array by di elements according to its sign"""
        if np.shape(np.imag(h)):
            temp_type = 'complex'
        else:
            temp_type = 'double'	
        
        if di>0:
            temp = np.zeros(len(h), dtype=temp_type)
            temp[:-di] = h[di:]
        elif di<0:
            temp = np.zeros(len(h)-di, dtype=temp_type)
            temp[-di:] = h
        else:
            temp = h
        return temp	
    
    def __call__(self, q_eval, Mtot, shiftQ=True):
        """Evaluate surrogate at sample rate modified by total mass"""

        # allocate memory for result of polynomial fits
        Amp_eval = np.zeros([self.dim_rb,1])
        phase_eval = np.zeros([self.dim_rb,1])
        
        # allocate memory for basis interpolants
        time_samples = int( (self.Mtot/Mtot) * self.time_samples )
        B_real = np.zeros([time_samples, self.dim_rb], dtype='double')
        B_imag = np.zeros([time_samples, self.dim_rb], dtype='double')
        
        # map specific q_eval to reference interval
        q_0 = 2*(q_eval - self.qmin)/(self.qmax - self.qmin) - 1;
        
        # If given Mtot is larger than fiducial Mtot then don't extrapolate interpolant
        #if Mtot <= self.Mtot:
        #    interp_times = np.arange(self.time_samples) * (Mtot/self.Mtot)*self.deltaT
        #else:
        #    interp_times = np.arange(int(self.Mtot/Mtot*self.time_samples)) * (Mtot/self.Mtot)*self.deltaT
        interp_times = np.arange(self.time_samples) * self.deltaT
        
        for jj in range(self.dim_rb):
            Amp_eval[jj] = np.polyval(self.AmpCoeff[jj,0:self.dim_rb], q_0)
            phase_eval[jj] = np.polyval(self.PhaseCoeff[jj,0:self.dim_rb], q_0)
            B_real[:len(interp_times),jj] = interpolate.splev(interp_times, (self.re_t[jj],self.re_c[jj],self.re_k[jj]))
            B_imag[:len(interp_times),jj] = interpolate.splev(interp_times, (self.im_t[jj],self.im_c[jj],self.im_k[jj]))
        h_EIM = Amp_eval*np.exp(-1j*phase_eval)
        
        self.Binterp = B_real + 1j*B_imag
        
        surrogate = np.dot(self.Binterp, h_EIM)
        hp = surrogate.real
        hc = surrogate.imag
        
        i_peak = 0
        if shiftQ:
            i_peak = self.argpeak(surrogate)
        times = np.arange(time_samples) * self.deltaT
        times -= times[i_peak]
        
        return times, hp, hc
    
    def plot_surrogate(self, q_eval, Mtot, hpQ=True, hcQ=False, shiftQ=True):
        """Plot plus and/or cross polarizations of a surrogate waveform"""
        leg = np.array(['$h_+$', '$h_\\times$'])
        plotQ = np.array([hpQ, hcQ], dtype='bool')
        times, hp, hc = self(q_eval, Mtot, shiftQ)
        if hpQ:
            plot(times, hp, 'r-')
        if hcQ:
            plot(times, hc, 'b-')
        xlabel('$t$ (sec)')
        legend(leg[plotQ])
        show()
    
    def plot_surrogates(self, qMtot, hpQ=True, hcQ=False, shiftQ=True):
        """Plot plus and/or cross polarizations of multiple surrogate waveforms"""
        leg = np.array(['$h_+$', '$h_\\times$'])
        plotQ = np.array([hpQ, hcQ], dtype='bool')
        for jj in range(np.shape(qMtot)[0]):
            times, hp, hc = self(qMtot[jj][0], qMtot[jj][1], shiftQ)
            if hpQ:
                plot(times, hp, 'r-')
            if hcQ:
                plot(times, hc, 'b-')
        xlabel('$t$ (sec)')
        legend(leg[plotQ])
        show()
