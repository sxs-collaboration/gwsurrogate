""" Gravitational Wave Surrogate classes for text and hdf5 files"""

from __future__ import division # for python 2

__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@umassd.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Jonathan Blackman, Scott Field, Chad Galley, Vijay Varma, Kevin Barkett"
__version__ = "0.9.5"
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

# adding "_" prefix to module names so they won't show up in gws tab completion
import numpy as _np
from gwtools.harmonics import sYlm as _sYlm
from gwtools import gwtools as _gwtools
from gwtools import gwutils as _gwutils
from gwsurrogate.new.surrogate import ParamDim as _ParamDim
from gwsurrogate.new.surrogate import ParamSpace as _ParamSpace


from . import catalog

from .new import surrogate as _new_surrogate
from .new import precessing_surrogate as _precessing_surrogate
from .simple_surrogates import surrogate as _simple_surrogate
from .simple_surrogates.surrogate import EvaluateSingleModeSurrogate, EvaluateSurrogate # for backwards compatibility

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

        import warnings

        ## Allow violations within this value.
        # Sometimes, chi can be 1+1e-16 due to machine precision limitations,
        # this will ignore such cases
        grace = 1e-14

        chiAmag = _np.linalg.norm(chiA0)
        chiBmag = _np.linalg.norm(chiB0)

        if not self.keywords['Precessing']:
            if (_np.linalg.norm(chiA0[:2]) > grace
                    or _np.linalg.norm(chiB0[:2]) > grace):
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
    chiA0:      Dimensionless spin vector of the heavier black hole, given in
                the coprecessing frame at reference epoch.
    chiB0:      Dimensionless spin vector of the lighter black hole, given in
                the coprecessing frame at reference epoch.

                The coprecessing frame is the minimal rotation frame of
                arXiv:1110.2965, where the z-axis of this frame always tracks
                the direction of the instantaneous orbital angular momentum.
                For nonprecessing models, the direction of orbital angular
                momentum is constant and this is the same as the inertial
                frame.

    M, dist_mpc: Either specify both M and dist_mpc or neither.
        M        :  Total mass (solar masses). Default: None.
        dist_mpc :  Distance to binary system (MegaParsecs). Default: None.

    f_low :     Instantaneous initial frequency of the (2, 2) mode. In
                practice, this is estimated to be twice the initial orbital
                frequency in the coprecessing frame.
                Should be in cycles/M if units = 'dimensionless', should be in
                Hertz if units = 'mks'.
                If 0, the entire waveform is returned.
                Default: None, must be specified by user.

                NOTE: For some models like NRSur7dq4, only f_low=0 is allowed.
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
                indicies for each ell will be included automatically.
                Default: None, in which case all available modes wll be
                included.

    mode_list : A list of (ell, m) modes tuples to be included.
                Example: mode_list = [(2,2),(2,1)].
                Default: None, in which case all available modes are included.
                The m<0 modes will automatically be included for nonprecessing
                models. At most one of ellMax and mode_list can be specified.

                Note: mode_list is allowed only for nonprecessing models; for
                precessing models use ellMax. For precessing systems, all m
                indices of a given ell index mix with each other, so there is
                no clear hierarchy. To get the individual modes just don't
                specify inclination and a dictionary of modes will be returned.

    phi_ref :   Orbital phase at reference epoch. Default: 0.

    inclination : Inclination angle between the orbital angular momentum
                direction at the reference epoch and the line-of-sight to the
                observer. If inclination is None, the mode data is returned as
                a dictionary. If specified, the complex strain (h = hplus -i
                hcross) evaluated at (inclination, pi/2) on the sky of the
                reference frame is returned. See below for definition of the
                reference frame. Default: None.

    precessing_opts:
                A dictionary containing optional parameters for a precessing
                surrogate model. Default: None.
                Allowed keys are:
                init_quat: The initial unit quaternion (length 4 vector)
                    giving the rotation from the coprecessing frame to the
                    inertial frame at the reference epoch.
                    Default: None, in which case the spins in the coprecessing
                    frame are equal to the spins in the inertial frame.
                return_dynamics:
                    Return the frame dynamics and spin evolution along with
                    the waveform. Default: False.
                use_lalsimulation_conventions:
                    If True, interprets the spin directions and init_orbphase
                    using lalsimulation conventions. Specifically, before
                    evaluating the surrogate, the spins will be rotated about
                    the z-axis by init_phase. Default: True (see 
                    DynamicsSurrogate, which is the only place this option is
                    used).
                Example: precessing_opts = {
                                    'init_quat': [1,0,0,0],
                                    'return_dynamics': True,
                                    'use_lalsimulation_conventions': True
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

    taper_end_durataion:
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
                    -i hcross) evaluated at (inclination, pi/2) on the sky of
                    the reference frame is returned. This follows the LAL
                    convention, see below for details.  This includes all modes
                    given in the ellMax/mode_list argument. For nonprecessing
                    systems the m<0 modes are automatically deduced from the
                    m>0 modes. To see if a model is precessing check
                    self.keywords.

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
                    The time-dependent inertial frame chiA with shape (L, 3)
                chiB = dynamics['chiB']
                    The time-dependent inertial frame chiB with shape (L, 3)


    IMPORTANT NOTES:
    ===============

    The reference frame (or inertial frame) is defined as follows:
        The +ve z-axis is along the orbital angular momentum at the reference
        epoch. The orbital phase at the reference epoch is phi_ref. This means
        that the separation vector from the lighter BH to the heavier BH is at
        an azimuthal angle phi_ref from the +ve x-axis, in the orbital plane at
        the reference epoch. The y-axis completes the right-handed triad. The
        reference epoch is set using f_ref.

        Now, if inclination is given, the waveform is evaluated at
        (inclination, pi/2) in the reference frame. This agrees with the LAL
        convention. See Harald Pfeiffer's, LIGO DCC document T18002260-v1 for
        the LAL frame diagram.
        """

        chiA0 = _np.array(chiA0)
        chiB0 = _np.array(chiB0)

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
        domain, h, dynamics = self._sur_dimless(x, phi_ref=phi_ref,
            fM_low=fM_low, fM_ref=fM_ref, dtM=dtM, timesM=timesM, dfM=dfM,
            freqsM=freqsM, mode_list=mode_list, ellMax=ellMax,
            precessing_opts=precessing_opts, tidal_opts=tidal_opts,
            par_dict=par_dict)

        # taper the last portion of the waveform, regardless of whether or not
        # this corresponds to inspiral, merger, or ringdown.
        if taper_end_duration is not None:
            h_tapered = {}
            for mode, hlm in h.iteritems():
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
            h = self._mode_sum(h, inclination, _np.pi/2,
                    fake_neg_modes=fake_neg_modes)

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
            if not _np.array_equal(domain, times):
                raise Exception("times were given as input but returned "
                    "domain somehow does not match.")
        if (freqs is not None):
            if not _np.array_equal(domain, freqs):
                raise Exception("freqs were given as input but returned "
                    "domain somehow does not match.")

        # Rescale waveform to physical units
        if amp_scale != 1:
            if type(h) == dict:
                h.update((x, y*amp_scale) for x, y in h.items())
            else:
                h *= amp_scale

        return domain, h, dynamics


class SpEC_q1_10_NoSpin(SurrogateEvaluator):
    """
A class for the SpEC_q1_10_NoSpin surrogate model presented in
http://arxiv.org/abs/1502.07758

Evaluates gravitational waveforms generated by nonspinning binary black hole
systems. This model was built using numerical relativity (NR) waveforms.

This model includes up to ell_max = 8 modes. The m<0 modes are deduced from
the m>0 modes.

The parameter space of validity is:
q \in [1, 10]
where q is the mass ratio. 

The surrogate has been trained in the range
q \in [1, 10]
and has been tested against existing NR waveforms in that range.

See the __call__ method on how to evaluate waveforms.
In the __call__ method, x must have format x = [q].
    """

    def __init__(self, h5filename):
        self.h5filename = h5filename
        domain_type = 'Time'
        keywords = {
            'Precessing': False,
            'Hybridized': False,
            }
        # soft_lims -> raise warning when outside lims
        # hard_lim -> raise error when outside lims
        # Format is [qMax].
        soft_param_lims = [10.01, 0.0]
        hard_param_lims = [10.01, 0.0]
        super(SpEC_q1_10_NoSpin, self).__init__(self.__class__.__name__, \
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
        sur = _simple_surrogate.SurrogateEvaluatorWrapper()
        #sur.load(self.h5filename)
        # TODO: Write Me!
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
        x = [q, 0, 0]
        return x



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
In the __call__ method, x must have format x = [q, chi1z, chi2z].
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
        sur = _new_surrogate.AlignedSpinCoOrbitalFrameSurrogate()
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
In the __call__ method, x must have format x = [q, chi1z, chi2z].
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
        sur = _new_surrogate.AlignedSpinCoOrbitalFrameSurrogateTidal()
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
In the __call__ method, x must have format x = [q, chi1, chi2].
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
        sur = _precessing_surrogate.PrecessingSurrogate(self.h5filename)
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
    "NRSur7dq4": NRSur7dq4,
    "NRHybSur3dq8Tidal": NRHybSur3dq8Tidal,
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


        import os

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

