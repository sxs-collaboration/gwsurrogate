""" Gravitational Wave Surrogate class """

from __future__ import division

__copyright__ = "Copyright (C) 2014 Scott Field"
__email__     = "sfield@umd.edu"
__status__    = "testing"
__author__    = "Scott Field"

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

import numpy as np
import const_mks as mks
from pylab import matplotlib as plt
import time

class SurrogateGW:
    def __init__(self, sdir):
        """initialize surrogate model by loading files from directory sdir"""

        ### Surrogate directory ####
        self.surrogate_dir = sdir

        ### Surrogate's sampling rate, mass ratio and total mass ###
        self.sample_rate      = np.loadtxt(sdir+'SampleRate.txt')
        self.q_interval       = np.loadtxt(sdir+'qRange.txt')
        self.Mtot             = np.loadtxt(sdir+'Mtot.txt')

        ### greedy points (ordered) ###
        self.greedypts = np.loadtxt(sdir+'greedy_q.txt')

        ### empirical time index ###
        self.eim_indx = np.loadtxt(sdir+'EIM_indx.txt')

        ### Complex B coefficients ###
        B_i    = np.loadtxt(sdir+'B_imag.txt')
        B_r    = np.loadtxt(sdir+'B_real.txt')
        self.B = B_r + (1j)*B_i

        ### Deduce sizes from B ###
        self.dim_rb       = B_r.shape[1]
        self.time_samples = B_r.shape[0]

        ### Coefficients for phase/amp parametric fit (polynomial) ###
        self.PhaseCoeff = np.loadtxt(sdir+'PhasePolyCoeff.txt')
        self.AmpCoeff   = np.loadtxt(sdir+'AmpPolyCoeff.txt')

    def __call__(self,q_eval):
        """evaluate surrogate at q_eval"""

        ### Allocate memory for polynomial fit evaluations ###
        Amp_eval   = np.zeros([self.dim_rb,1])
        phase_eval = np.zeros([self.dim_rb,1])

        ### Map q_eval to the standard interval ###
        qmin = self.q_interval[0]
        qmax = self.q_interval[1]
        q_0 = 2*(q_eval - qmin)/(qmax - qmin) - 1;

        ### Evaluate fits ###
        for jj in range(self.dim_rb):
            Amp_eval[jj]     = np.polyval(self.AmpCoeff[jj,0:self.dim_rb],q_0)
            phase_eval[jj]   = np.polyval(self.PhaseCoeff[jj,0:self.dim_rb],q_0)

        ### Build dim_RB-vector fit evalution of h ###
        h_EIM = Amp_eval*np.exp(-1j*phase_eval)

        ### Surrogate modes hp and hc ###
        surrogate = np.dot(self.B,h_EIM)
        hp        = surrogate.real
        hp        = hp.reshape([self.time_samples,])
        hc        = surrogate.imag
        hc        = hc.reshape([self.time_samples,])

        times = np.arange(self.time_samples) / self.sample_rate

        return times, hp, hc

    def plot(self,q_eval):
        """plot surrogate evaluated at q_eval"""

        times, hp, hc = self(q_eval)
        plt.pyplot.plot(times,hp)
        plt.pyplot.hold
        plt.pyplot.plot(times,hc)
        plt.pyplot.legend(['h plus', 'h cross'])
        plt.pyplot.show()

    def h_rb(self,i):
        """generate the ith reduced basis waveform"""

        times, hp, hc = self(self.greedypts[i])

        return times, hp, hc

    def plot_rb(self,i):
        """plot the ith reduced basis waveform"""

        self.plot(self.greedypts[i])

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

    def writetxt(self,q_eval,filename='output'):
        """write waveform to text file"""

        times, hp, hc = self(q_eval)
        np.savetxt(filename,[times, hp, hc])

    def writebin(self,q_eval,filename='output.txt'):
        """write waveform to numpy binary file"""

        times, hp, hc = self(q_eval)
        np.save(filename,[times, hp, hc])
