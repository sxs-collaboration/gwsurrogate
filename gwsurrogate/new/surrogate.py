""" Gravitational Wave Surrogate classes for text and hdf5 files"""

from __future__ import division  # for py2

__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@astro.cornell.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Jonathan Blackman, Scott Field, Chad Galley, Vijay Varma, Kevin Barkett"

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
from scipy.interpolate import InterpolatedUnivariateSpline as _iuspline
from gwtools.harmonics import sYlm as _sYlm

if __package__ is "" or "None": # py2 and py3 compatible
  print("setting __package__ to gwsurrogate.new so relative imports work")
  __package__="gwsurrogate.new"

# assumes unique global names
from .saveH5Object import SimpleH5Object
from .saveH5Object import H5ObjectList
from .saveH5Object import H5ObjectDict
from .nodeFunction import NodeFunction
from .spline_evaluation import TensorSplineGrid, fast_complex_tensor_spline_eval
from gwsurrogate import spline_interp_Cwrapper
from .tidal_functions import UniversalRelationLambda2ToI, \
    UniversalRelationLambda2ToOmega2, UniversalRelationLambda2ToLambda3, \
    UniversalRelationLambda3ToOmega3, UniversalRelationLambda2ToAqm, \
    EffectiveDeformabilityFromDynamicalTides, PNT2Tidal, \
    EffectiveDissipativeDynamicalTides, StrainTidalEnhancementFactor

PARAM_NUDGE_TOL = 1.e-12 # Default relative tolerance for nudging edge cases


def _identity(r1, r2):
    return r1, r2
def _amp_phase(r1, r2):
    return r1['amp']*np.exp(1.j*r1['phase'])
def _re_im(r1, r2):
    return r1['re'] + 1.j*r1['im']

RECOMBINATION_FUNCS = {
    'identity': _identity,
    'amp_phase': _amp_phase,
    're_im': _re_im,
        }


def _mode_sum(modes, theta, phi):
    h = 0.
    for (ell, m), h_mode in modes.items(): # inefficient in py2
        h += _sYlm(-2, ell, m, theta, phi) * h_mode
    return h


def _splinterp(xout, xin, yin, k=3, ext='const'):
    """Uses InterpolatedUnivariateSpline to interpolate real or complex data"""
    if np.iscomplexobj(yin):
        re = _splinterp(xout, xin, np.real(yin), k=k, ext=ext)
        im = _splinterp(xout, xin, np.imag(yin), k=k, ext=ext)
        return re + 1.j*im
    else:
        return _iuspline(xin, yin, k=k, ext=ext)(xout)

def _splinterp_Cwrapper(xout, xin, yin):
    """Uses gsl splines with a wrapper to interpolate real or complex data.
    Uses natural boundary conditions instead of not-a-knot boundary conditions
    like InterpolatedUnivariateSpline."""
    if len(xin) != len(yin):
        raise Exception('Expected x and y input lengths to match.')
    if np.iscomplexobj(yin):
        re = _splinterp_Cwrapper(xout, xin, np.real(yin))
        im = _splinterp_Cwrapper(xout, xin, np.imag(yin))
        return re + 1.j*im
    else:
        return spline_interp_Cwrapper.interpolate(xout, xin, yin)


class ParamDim(SimpleH5Object):
    """
    A helper class containing the information and functions for a single
    parameter space dimension
    """

    def __init__(self, name='', min_val=0, max_val=1, rtol=PARAM_NUDGE_TOL):
        """
        name: A descriptive name for this parameter dimension
        min_val: The minimum allowed value for this parameter
        max_val: The maximum allowed value for this parameter
        rtol: A relative tolerance for nudging parameters lying outside
             [min_val, max_val]. Scaled by (max_val - min_val) for an absolute
             tolerance. Useful to avoid machine precision issues.
        """
        super(ParamDim, self).__init__()

        tol = rtol * (max_val - min_val)

        if min_val + 2*tol > max_val:
            raise Exception("tol %s is too large for %s with range [%s, %s]"%(
                            tol, name, min_val, max_val))

        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.tol = tol
        self.tol_max = max_val - tol
        self.tol_min = min_val + tol

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s: [%s, %s] with tol %s'%(self.name, self.min_val,
                                           self.max_val, self.tol)

    def nudge(self, x):
        """
        Returns a nudged version of x lying within [min_val+tol, max_val-tol].
        x can have any shape. Returns x if it already lies in the interval.
        x must already lie within [min_val - tol, max_val + tol].
        """
        xmax = np.max(x)
        xmin = np.min(x)

        if xmax > self.tol_max:
            if xmax > self.max_val + self.tol:
                raise Exception("The maximum allowed %s is %s, got %s"%(
                                self.name, self.max_val, xmax))
            x = np.amin([x, np.ones(np.shape(x)) * self.tol_max], axis=0)

        if xmin < self.tol_min:
            if xmin < self.min_val - self.tol:
                raise Exception("The maximum allowed %s is %s, got %s"%(
                                self.name, self.min_val, xmin))
            x = np.amax([x, np.ones(np.shape(x)) * self.tol_min], axis=0)

        return x


class ParamSpace(SimpleH5Object):
    """
    A helper class for all the parameter domain related information and
    functions.
    """

    def __init__(self, name='', params=[]):
        """
        name: A descriptive name for this parameter space
        params: A list of ParamDim instances, one per parameter dimension
        """
        super(ParamSpace, self).__init__(['name', 'dim'], ['_params'])

        self.name = name
        self._params = H5ObjectList(params)
        self.dim = len(params)

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s: %s'%(self.name, [p.name for p in self._params])

    def param_names(self):
        """Returns the list of names for each parameter space dimension"""
        return [p.name for p in self._params]

    def min_vals(self):
        """Returns a list of minimum parameter space values"""
        return [p.min_val for p in self._params]

    def max_vals(self):
        """Returns a list of maximum parameter space values"""
        return [p.max_val for p in self._params]

    def nudge_params(self, x):
        """
        Nudges parameters lying slightly outside the valid domain to the
        boundary. x can be a single 1d parameter vector or a 2d vector with
        shape (n_params, self.dim).
        """

        xshape = np.shape(x)

        # It's convenient to be able to accept a float instead of a length-1
        # array for 1d parameter spaces.
        if len(xshape) == 0:
            x = np.array([x])
            xshape = np.shape(x)

        if len(xshape) == 1:
            if len(x) != self.dim:
                raise Exception("Parameter space has dimension %s, got %s."%(
                                self.dim, xshape))
            res = np.array([p.nudge(xi) for p, xi in zip(self._params, x)])

        elif len(xshape) == 2:
            if xshape[1] != self.dim:
                raise Exception("Expecting array with shape (n, %s), got %s"%(
                                self.dim, xshape))
            res = np.array([p.nudge(xi) for p, xi in zip(self._params, x.T)])

        else:
            raise Exception("x should be 1d or 2d, got shape {}".format(xshape))

        return res

    def h5_prepare_subs(self):
        """Setup dummy subordinates before loading them"""
        params = [ParamDim() for _ in range(self.dim)]
        self._params = H5ObjectList(params)


class _SingleFunctionSurrogate_NoChecks(SimpleH5Object):
    """
    A surrogate model for a single (real or complex) function on a 1d domain.
    Skips sanity/validity checks, so should be called with sanitized inputs.
    Use SingleFunctionSurrogate for actual surrogates of single functions.
    """

    def __init__(self, name=None, ei_basis=None, node_functions=[]):
        """
        name: A descriptive name for the function this surrogate models
        ei_basis: A basis with shape (n_nodes, len(domain)).
                  Interpolation is done via nodes.dot(ei_basis)
        node_functions: A list of evaluators for each node.
                        Each one takes a parameter space vector x and returns
                        the node evaluated at x.
        """
        super(_SingleFunctionSurrogate_NoChecks, self).__init__(
                data_keys=['name', 'ei_basis', 'n_nodes'],
                sub_keys=['node_functions'])

        self.name = name
        self.ei_basis = ei_basis
        self.n_nodes = len(node_functions)
        self.node_functions = H5ObjectList(node_functions)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __call__(self, x):
        """
        Evaluates the surrogate at x, returning the result.
        """
        nodes = np.array([nf(x) for nf in self.node_functions])
        return nodes.dot(self.ei_basis)

    def h5_prepare_subs(self):
        """Setup NodeFunctions before loading them"""
        tmp_nodes = [NodeFunction() for _ in range(self.n_nodes)]
        self.node_functions = H5ObjectList(tmp_nodes)


class SingleFunctionSurrogate(_SingleFunctionSurrogate_NoChecks):
    """
    A surrogate model for a single (real or complex) function on a 1d domain.
    """

    def __init__(self, name=None, domain=None, param_space=None,
                 ei_basis=None, node_functions=[]):
        """
        name: A descriptive name for the function this surrogate models
        domain: A 1d array of the monotonically increasing domain
                (time/frequency) values
        param_space: A ParamSpace for this surrogate
        ei_basis: A basis with shape (n_nodes, len(domain)).
                  Interpolation is done via nodes.dot(ei_basis)
        node_functions: A list of evaluators for each node.
                        Each one takes a parameter space vector x and returns
                        the node evaluated at x.
        """
        super(SingleFunctionSurrogate, self).__init__(name=name,
                ei_basis=ei_basis, node_functions=node_functions)

        if domain is not None and np.min(np.diff(domain)) <= 0.0:
            raise Exception("domain should be monotonically increasing")

        self.domain = domain
        self.param_space = param_space

        self._h5_data_keys.append('domain')
        self._h5_subordinate_keys.append('param_space')

    def h5_prepare_subs(self):
        super(SingleFunctionSurrogate, self).h5_prepare_subs()
        self.param_space = ParamSpace()

    def __repr__(self):
        return '%s (%s)'%(self.name, self.param_space.name)

    def __call__(self, x, domain=None):
        """
        Evaluates the surrogate at x, returning the result.
        domain: An optional 1d array of domain values. If given, the result is
                evaluated at the domain values.
        """
        # Verify valid parameters and domain
        x = self.param_space.nudge_params(x)
        if domain is not None:
            if domain[0] < self.domain[0] or domain[-1] > self.domain[-1]:
                raise Exception("Domain must lie in [%s, %s]"%(
                                self.domain[0], self.domain[-1]))

        res = super(SingleFunctionSurrogate, self).__call__(x)

        if domain is not None:
            res = _splinterp(domain, self.domain, res)

        return res


class _ManyFunctionSurrogate_NoChecks(SimpleH5Object):
    """
    A container for surrogates sharing the same parameter space and domain.
    Skips checks, assuming they have already been performed.
    """

    def __init__(self, name='', single_function_components={},
                 many_function_components={}, combine_func='identity'):
        """
        name: A descriptive name for this surrogate.
        single_function_components: A dictionary of components, where the
                values are (ei_basis, node_functions) tuples.
        many_function_components: A dictionary of _ManyFunctionSurrogate_NoCheck
                components, where the values are
                (combine_func, single_function_components,
                 many_function_components) tuples.
        combine_func: A key of RECOMBINATION_FUNCS used to combine subordinate
                      results into a single result.
        """
        super(_ManyFunctionSurrogate_NoChecks, self).__init__(
                data_keys=['name', 'func_keys', 'sur_keys', 'combine_func'],
                sub_keys=['func_subs', 'sur_subs'])

        self.name = name
        self.combine_func = combine_func
        self.func_keys = list(single_function_components.keys())
        self.sur_keys = list(many_function_components.keys())
        if len(set(self.func_keys).intersection(self.sur_keys)) > 0:
            raise Exception("Component keys must be unique! Got %s, %s"%(
                            self.func_keys, self.sur_keys))

        func_sub_dict = {}
        for k, (ei, nf) in single_function_components.items(): # inefficient in py2
            func_sub_dict[k] = _SingleFunctionSurrogate_NoChecks(k, ei, nf)
        self.func_subs = H5ObjectDict(func_sub_dict)

        sur_sub_dict = {}
        for k, (cf, sfc, mfc) in many_function_components.items(): # inefficient in py2
            sur_sub_dict[k] = _ManyFunctionSurrogate_NoChecks(k, sfc, mfc, cf)
        self.sur_subs = H5ObjectDict(sur_sub_dict)

    def h5_prepare_subs(self):
        """
        Initialize subordinate surrogate class instances so they can
        load their own data from h5 files.
        """
        self.func_subs = H5ObjectDict({k: _SingleFunctionSurrogate_NoChecks()
                                       for k in self.func_keys})
        self.sur_subs = H5ObjectDict({k: _ManyFunctionSurrogate_NoChecks()
                                      for k in self.sur_keys})

    def __str__(self):
        return self.name

    def __call__(self, x):
        func_evals = {k: sur(x) for k, sur in self.func_subs.iteritems()} # inefficient in py2
        sur_evals = {k: sur(x) for k, sur in self.sur_subs.iteritems()} # inefficient in py2
        return RECOMBINATION_FUNCS[self.combine_func](func_evals, sur_evals)

    def _eval_func(self, x, key):
        return self.func_subs[key](x)

    def _eval_sur(self, x, key):
        return self.sur_subs[key](x)


class ManyFunctionSurrogate(_ManyFunctionSurrogate_NoChecks):
    """
    A container for surrogates sharing the same parameter space and domain.
    """

    def __init__(self, name='', domain=None, param_space=None,
                 single_function_components={}, many_function_components={},
                 combine_func='identity'):
        """
        name: A descriptive name for this surrogate.
        param_space: A ParamSpace for this surrogate.
        single_function_components: A dictionary of components, where the
                values are (ei_basis, node_functions) tuples.
        many_function_components: A dictionary of _ManyFunctionSurrogate_NoCheck
                components, where the values are
                (combine_func, single_function_components,
                 many_function_components) tuples.
        combine_func: A key of RECOMBINATION_FUNCS used to combine subordinate
                      results into a single result.
        """
        super(ManyFunctionSurrogate, self).__init__(
                name=name,
                single_function_components=single_function_components,
                many_function_components=many_function_components,
                combine_func=combine_func,
            )

        self.domain = domain
        self.param_space = param_space
        self._h5_data_keys.append('domain')
        self._h5_subordinate_keys.append('param_space')

    def h5_prepare_subs(self):
        super(ManyFunctionSurrogate, self).h5_prepare_subs()
        self.param_space = ParamSpace()

    def __call__(self, x):
        """
        Evaluates the surrogate at x, returning the result.
        """
        # Verify valid parameters
        x = self.param_space.nudge_params(x)
        return super(ManyFunctionSurrogate, self)(x)

class FastTensorSplineSurrogate(SimpleH5Object):
    """
    A special case of having a complex empirical interpolant combined with
    tensor splines for the real and imaginary parts of each empirical node,
    for each waveform mode. All tensor splines must use the same grid.
    Written for speed, minimizing python operations
    between tensor spline interpolations, which are done simultaneously for
    each mode to keep numpy busy. Obtained ~25ms evaluation time per waveform
    for 12 waveform modes. This was ~66ms when using the python class
    hierarchy, and using a separate call to numpy for each tensor spline
    interpolation. Note that similar C code written with gsl splines takes
    ~50ms, but should have room for optimization.
    """

    def __init__(self, name=None, domain=None, param_space=None,
                 knot_vecs=[], mode_data={}, modes=None):

        super(FastTensorSplineSurrogate, self).__init__(
                sub_keys=['param_space', 'ts_grid'],
                data_keys=['name', 'domain', 'ei', 'cre', 'cim', 'mode_list',
                           'mode_indices'])

        self.name = name
        self.domain = domain
        self.param_space = param_space
        if param_space is None:
            self.param_space = ParamSpace()
        if modes is None:
            modes = list(mode_data.keys())
        self.mode_list = modes
        self.mode_indices = {str(k): i for i, k in enumerate(modes)}
        self.ei = [mode_data[k][0] for k in modes]
        self.cre = [mode_data[k][1] for k in modes]
        self.cim = [mode_data[k][2] for k in modes]
        self.ts_grid = TensorSplineGrid(knot_vecs)

    def __call__(self, x, theta=None, phi=None, modes=None):
        """
        Return surrogate evaluation.
        Arguments:
            x : The intrinsic parameters (see self.param_space)
            theta/phi : polar and azimuthal angles of the direction of
                        gravitational wave emission. If given, sums up modes
                        and returns h_plus and h_cross (default returns modes)
            modes : A list of (ell, m) modes to be evaluated (default: all)
        Returns h:
            h : If theta and phi are None, h is a dictionary of waveform modes
                sampled at self.domain with (ell, m) keys.
                If theta and phi are given, h = h_plus - i * h_cross is a
                complex array given by the sum of the modes.
        """
        if (theta is None) != (phi is None):
            raise Exception("Either give theta and phi or neither")

        x = self.param_space.nudge_params(x)

        if modes is None:
            modes = self.mode_list

        h_modes = {}
        for k in modes:
            i = self.mode_indices[str(k)]

            h_eim = fast_complex_tensor_spline_eval(x,self.ts_grid,self.cre[i],self.cim[i])

            # Evaluate the empirical interpolant
            h_modes[k] = h_eim.dot(self.ei[i])

        if theta is not None:
            return _mode_sum(h_modes, theta, phi)

        return h_modes


class MultiModalSurrogate(ManyFunctionSurrogate):
    """
    A surrogate for multimodal waveforms, where each waveform mode has
    its own surrogate. Contains added functionality for evaluating on
    sphere.
    """

    def __init__(self, name=None, domain=None, param_space=None,
                 mode_data={}, mode_type='complex', modes=None):
        """
        name: A descriptive name for this surrogate.
        domain: A 1d array of the monotonically increasing domain
                (time/frequency) values
        param_space: A ParamSpace for this surrogate.
        mode_data: A dictionary of modes with (l, m) integer keys,
                   where the values are (ei_basis, node_functions) tuples.
        mode_type: Can be 'amp_phase', 're_im', or 'complex' depending on
                   how mode surrogates are built. If 'amp_phase' or 're_im',
                   mode_data values should instead be dictionaries with keys
                   ['amp', 'phase'] or ['re', 'im'] and
                   (ei_basis, node_functions) values.
        modes: A list of (ell, m) modes giving an ordering to mode_data.keys().
               If None, uses mode_data.keys().
        """

        if mode_type == 'complex':
            super(MultiModalSurrogate, self).__init__(name, domain, param_space,
                                                      mode_data, {}, 'identity')
        elif mode_type in ['amp_phase', 're_im']:
            mode_data = {k: (mode_type, v, {})
                         for k, v in mode_data.items()} # inefficient in py2
            super(MultiModalSurrogate, self).__init__(name, domain, param_space,
                                                      {}, mode_data, mode_type)
        else:
            raise ValueError("Invalid mode_type: %s"%(mode_type))

        self.mode_type = mode_type
        if modes is None:
            self.modes = list(mode_data.keys())
        else:
            self.modes = modes

        self._h5_data_keys.append('modes')
        self._h5_data_keys.append('mode_type')


    def __call__(self, x, theta=None, phi=None, modes=None):
        """
        Return surrogate evaluation.
        Arguments:
            x : The intrinsic parameters (see self.param_space)
            theta/phi : polar and azimuthal angles of the direction of
                        gravitational wave emission. If given, sums up modes
                        and returns h_plus and h_cross (default returns modes)
            modes : A list of (ell, m) modes to be evaluated (default: all)
        Returns h:
            h : If theta and phi are None, h is a dictionary of waveform modes
                sampled at self.domain with (ell, m) keys.
                If theta and phi are given, h = h_plus - i * h_cross is a
                complex array given by the sum of the modes.
        """
        if (theta is None) != (phi is None):
            raise Exception("Either give theta and phi or neither")

        x = self.param_space.nudge_params(x)

        if modes is None:
            modes = self.modes

        if self.mode_type == 'complex':
            h_modes = {k: self._eval_func(x, k) for k in modes}
        else:
            h_modes = {k: self._eval_sur(x, k) for k in modes}

        if theta is not None:
            return _mode_sum(h_modes, theta, phi)

        return h_modes


class AlignedSpinCoOrbitalFrameSurrogate(ManyFunctionSurrogate):
    """
    A surrogate for coorbital frame multimodal waveforms, where each waveform
    data piece has its own surrogate.

    The waveform data pieces are:
    Amplitude and phase of the (2,2) mode.
    Real and imaginary parts of coorbital frame waveform for other modes.
    """

    def __init__(self, name=None, domain=None, param_space=None, \
            phaseAlignIdx=None, TaylorT3_t_ref=None, \
            coorb_mode_data={(2, 2): {}}
            ):
        """
        name:               A descriptive name for this surrogate.

        domain:             A 1d array of the monotonically increasing time
                            values.

        param_space:        A ParamSpace for this surrogate.

        phaseAlignIdx:      This value should be loaded directly from the
            surrogate's h5 file. Index of domain at which the orbital phase is
            aligned. This is used when putting back the TaylorT3 contribution
            that was subtracted before modeling the phase.

        TaylorT3_t_ref:     This value should be loaded directly from the
            surrogate's h5 file. This is an arbitrary reference time used
            in the TaylorT3 contribution, but is fixed during the surrogate
            construction.

        coorb_mode_data: A dictionary of modes with (l, m) integer keys, where
            the values are themselves dictionaries containing the coorbital
            frame waveform for that mode. The coorbital frame is defined as:
            H_lm = h_lm*exp(i*m*phi_22/2), where h_lm is the inertial frame
            waveform and h_22 = A_22 * exp(-i phi_22). NOTE the minus sign.

            coorb_mode_data should be a dict with mode_key: mode_value pairs,
                where mode_value = AmpPhase_22 for mode_key = (2, 2)
                and mode_value = CoorbReIm_lm for other modes.

            Above, AmpPhase_22 and CoorbReIm_lm are themselved dictionaries:
                AmpPhase_22 = {'amp': Amp_22, 'phase': phi_22}
                CoorbReIm_lm = {'re': re_H_lm, 'im': im_H_lm}, where
                re_H_lm = Real(H_lm) and im_H_lm = Imag(H_lmc).
            Finally, all of Amp_22, phi_22, re_H_lm and im_H_lm should be
            (ei_basis, node_functions) tuples of that data piece.

            IMPORTANT NOTE: The phase of (2, 2) mode should be defined with
            a minus sign as shown above, this is opposite to what is done for
            MultiModalSurrogate.
        """

        # get list of modes, but move (2,2) mode to start of the list.
        # This is important because we need the phase of the 22 mode to
        # transform the other modes from coorbital frame to inertial frame.
        self.mode_list = list(coorb_mode_data.keys())
        mode22_idx = [i for i in range(len(self.mode_list)) \
            if self.mode_list[i] == tuple([2, 2])]
        if len(mode22_idx) != 1:
            raise Exception('Seems to have found multiple or no 22 mode!')
        mode22_idx = mode22_idx[0]

        # shift 22 mode to the first index
        self.mode_list.insert(0, self.mode_list.pop(mode22_idx))

        if self.mode_list[0] != tuple([2, 2]):
            raise Exception('Expected the first mode at this point to be the'\
                ' 22 mode.')
        # make sure shifting the 22 mode index did not delete or add a
        # mode by mistake
        if len(self.mode_list) != len(coorb_mode_data.keys()):
            raise Exception('Number of modes do not agree')

        self.mode_type = 'identity'
        many_function_components = {}
        for mode in self.mode_list:
            many_function_components[mode] = ('identity', \
                coorb_mode_data[mode], {})

        # required for TaylorT3
        self.phaseAlignIdx = phaseAlignIdx
        self.TaylorT3_t_ref = TaylorT3_t_ref
        self.TaylorT3_factor_without_eta = None

        super(AlignedSpinCoOrbitalFrameSurrogate, self).__init__(name,
                domain, param_space, {}, many_function_components,
                self.mode_type)

        self._h5_data_keys.append('mode_list')
        self._h5_data_keys.append('mode_type')
        self._h5_data_keys.append('phaseAlignIdx')
        self._h5_data_keys.append('TaylorT3_t_ref')

    def _search_omega(self, omega22, omega_val):
        """ Find closest index such taht omega22[index] = omega_val
        """
        # find first index where omega22 > omega_val
        idx = np.where(omega22 > omega_val)[0][0]
        # if idx-1 is closer to omega_val, pick that instead
        if abs(omega22[idx-1] - omega_val) < abs(omega22[idx] - omega_val):
            idx -= 1
        return idx

    def _coorbital_to_inertial_frame(self, h_coorb, h_22, mode_list, dtM,
        timesM, fM_low, fM_ref, do_not_align):
        """ Transforms a dict from Coorbital frame to inertial frame.

            The surrogate data is sparsely sampled, so upsamples to time
            step dtM if given. This is done in the coorbital frame since
            the waveform is slowly varying in that frame.

            If fM_low is given, only part of the waveform where frequency of
            the (2, 2) mode is greater than fM_low is retained.

            if do_not_align = False:
                Aligns the 22 mode phase to be 0 at fM_ref. This means
                that at this reference frequency, the heavier BH is roughly on
                the +ve x axis and the lighter BH is on the -ve x axis.
            do_not_align should be True only when converting from pySurrogate
            format to gwsurrogate format as we may want to do some checks that
            the waveform has not been modified
        """

        Amp_22 = h_22[0]['amp']
        phi_22 = h_22[0]['phase']
        domain = np.copy(self.domain)

        # Get omega22_sparse, the angular frequency of the 22 mode, from the
        # sparse surrogate domain.
        # Use np.diff instead of np.gradient to match the LAL version
        omega22_sparse = np.append(np.diff(phi_22)/np.diff(domain), 0)

        # t=0 is at the waveform peak for the surrogate
        peak22Idx = np.argmin(np.abs(domain))
        omega22_peak = omega22_sparse[peak22Idx]
        # We ignore the part after the peak.  This way we avoid the noisy part
        # at late times, which can randomly be at frequency = fM_low.
        omega22_sparse = omega22_sparse[domain <= domain[peak22Idx]]

        # Get initIdx such that the initial (2, 2) mode frequency ~ fM_low.
        # We will make this more precise below.
        if fM_low != 0:
            omega_low = 2*np.pi*fM_low
            if omega_low < omega22_sparse[0]:
                raise ValueError('f_low is lower than the minimum allowed'
                    ' frequency')
            if omega_low > omega22_peak:
                raise ValueError('f_low is higher than the peak frequency')

            # Choose 5 indices less, to ensure omega_low is included
            initIdx = self._search_omega(omega22_sparse, omega_low) - 5
            # But if initIdx < 0, we are at the start of the surrogate data
            # so just choose 0
            if initIdx < 0:
                initIdx = 0

        else:
            # If fM_low is 0, we use the entire waveform
            initIdx = 0

            # But, if fM_low = 0 and timesM is given, the output of the
            # interpolant depends very slightly on the length of the sparse
            # data used to construct the interpolant. So, to achieve machine
            # precision equivalence between using dtM and timesM options, we
            # need to do the following: truncate before interpolation to the
            # same index as the dtM option would have used above (initIdx).
            # Using 6 rather than 5 because of the greater than condition.
            if timesM is not None:
                initIdx = np.where(domain > timesM[0])[0][0] - 6

        Amp_22 = Amp_22[initIdx:]
        phi_22 = phi_22[initIdx:]
        domain = domain[initIdx:]

        if timesM is not None:
            if timesM[-1] > domain[-1]:
                raise Exception("'times' includes times larger than the"
                    " maximum time value in domain.")
            if timesM[0] < domain[0]:
                raise Exception("'times' starts before start of domain. Try"
                    " increasing initial value of times or reducing f_low.")

        if dtM is None and timesM is None:
            # Use the sparse domain
            timesM = domain
            omega22 = omega22_sparse[initIdx:]
            do_interp = False
        else:
            ## Interpolate onto uniform-domain/timesM if needed
            do_interp = True
            if dtM is not None:
                t0 = domain[0]
                tf = domain[-1]
                num_times = int(np.ceil((tf - t0)/dtM));
                timesM = t0 + dtM*np.arange(num_times)
            else:
                if timesM[0] < domain[0] or timesM[-1] > domain[-1]:
                    raise Exception('Trying to evaluate at times outside the'
                        ' domain.')

            Amp_22 = _splinterp_Cwrapper(timesM, domain, Amp_22)
            phi_22 = _splinterp_Cwrapper(timesM, domain, phi_22)

            # now recompute omega22 with the dense data, but retain only data
            # upto the peak to avoid the noisy part
            omega22 = np.append(np.diff(phi_22)/np.diff(timesM), 0)
            omega22 = omega22[timesM <= 0]

            # Truncate data so that only freqs above omega_low are retained
            # If timesM are already given, we don't need to truncate data
            if dtM is not None:
                if fM_low != 0:
                    startIdx = self._search_omega(omega22, omega_low)
                else:
                    # If fM_low is 0, we use the entire waveform
                    startIdx = 0

                Amp_22 = Amp_22[startIdx:]
                phi_22 = phi_22[startIdx:]
                omega22 = omega22[startIdx:]
                timesM = timesM[startIdx:]


        # Get reference index where waveform needs to be aligned.
        if (abs(fM_ref-fM_low) < 1e-13) and (dtM is not None):
            # This means that the data is already truncated at fM_low,
            # so we just need the first index for fM_ref=fM_low
            refIdx = 0
        else:
            omega_ref = 2*np.pi*fM_ref
            if omega_ref > omega22_peak:
                raise ValueError('f_ref is higher than the peak frequency')

            refIdx = self._search_omega(omega22, omega_ref)


        # do_not_align should be True only when converting from pySurrogate
        # format to gwsurrogate format as we may want to do some checks that
        # the waveform has not been modified
        if not do_not_align:
            # Set orbital phase to 0 refIdx. Note that the Coorbital
            # frame data is not affected by this constant phase shift.

            # The orbital phase is obtained as phi_22/2, so this leaves a pi
            # ambiguity.  But the surrogate data is already aligned such that
            # the heavier BH is on the +ve x-axis at t=-1000M. See Sec.VI.A.4
            # of arxiv:1812.07865, the resolves the pi ambiguity. This means
            # that the after the realignment, the orbital phase at reference
            # frequency is 0.
            phi_22 += -phi_22[refIdx]

        h_dict = {}
        for mode in mode_list:
            if mode == tuple([2, 2]):
                h_dict[mode] = Amp_22 * np.exp(-1j*phi_22)
            else:
                l,m = mode
                h_coorb_lm = 0
                if 're' in h_coorb[mode][0].keys():
                    h_coorb_lm += h_coorb[mode][0]['re'] + 1j * 0
                if 'im' in h_coorb[mode][0].keys():
                    h_coorb_lm += 1j*h_coorb[mode][0]['im']

                h_coorb_lm = h_coorb_lm[initIdx:]
                if do_interp:
                    h_coorb_lm = _splinterp_Cwrapper(timesM,domain,h_coorb_lm)

                h_dict[mode] = h_coorb_lm * np.exp(-1j*m*phi_22/2.)

        return timesM, h_dict, None     # None is for dynamics

    def _set_TaylorT3_factor(self):
        """ Sets a term used in the 0 PN TaylorT3 phase. See Eq.43 of
        arxiv.1812.07865.
        """
        # Set only once
        if self.TaylorT3_factor_without_eta is None:
            # TaylorT3_t_ref is arbitrary. This is where the phase diverges,
            # so we choose it much after ringdown. This matches what was used
            # in the construction of the surrogate. See discussion near Eq.43
            # of arxiv.1812.07865
            theta_without_eta = ((self.TaylorT3_t_ref -self.domain)/5)**(-1./8)
            self.TaylorT3_factor_without_eta = -2./theta_without_eta**5

    def _TaylorT3_phase_22(self, x):
        """ 0 PN TaylorT3 phase. See Eq.43 of arxiv.1812.07865
        """

        q = x[0]
        eta = q/(1.+q)**2

        # 0PN TaylorT3 phase
        phi22_T3 = 1./eta**(3./8) * self.TaylorT3_factor_without_eta

        # Align at phaseAlignIdx
        phi22_T3 -= phi22_T3[self.phaseAlignIdx]

        return phi22_T3


    def __call__(self, x, fM_low=None, fM_ref=None, dtM=None,
            timesM=None, dfM=None, freqsM=None, mode_list=None, ellMax=None,
            precessing_opts=None, tidal_opts=None, par_dict=None,
            return_dynamics=False, do_not_align=False):
        """
    Return dimensionless surrogate modes.
    Arguments:
    x :             The intrinsic parameters EXCLUDING total Mass (see
                    self.param_space)

    fM_low :        Initial frequency of (2,2) mode in units of cycles/M.
                    If 0, will use the entire data of the surrogate.
                    Default None.

    fM_ref:         Frequency used to set the reference epoch at which
                    the reference frame is defined and the spins are specified.
                    See below for definition of the reference frame.
                    Default: None.

                    For time domain models, f_ref is used to determine a t_ref,
                    such that the frequency of the (2, 2) mode equals f_ref at
                    t=t_ref.

    dtM :           Uniform time step to use, in units of M. If None, the
                    returned time array will be the array used in the
                    construction of the surrogate, which can be nonuniformly
                    sampled.
                    Default None.

    timesM:         Time samples to evaluate the waveform at. Use either dtM or
                    timesM, not both.

    dfM :           This should always be None as for now we are assuming
                    a time domain model.

    freqsM:         Frequency samples to evaluate the waveform at. Use either
                    dfM or freqsM, not both.

    ellMax:         Maximum ell index for modes to include. All available m
                    indicies for each ell will be included automatically.
                    Default: None, in which case all available modes wll be
                    included.

    mode_list :     A list of (ell, m) modes to be evaluated.
                    Default None, which evaluates all avilable modes.
                    Will deduce the m<0 modes from m>0 modes.

    par_dict:       This should always be None for this model.

    do_not_align:   Ignore fM_ref and do not align the waveform. This should be
                    True only when converting from pySurrogate format to
                    gwsurrogate format as we may want to do some checks that
                    the waveform has not been modified.

    Returns
    timesM, h, dynamics:
        timesM : time array in units of M.
        h : A dictionary of waveform modes sampled at timesM with
            (ell, m) keys.
        dynamics: None, since this is a nonprecessing model.


    IMPORTANT NOTES:
    ===============

    The reference frame (or inertial frame) is defined as follows:
        The +ve z-axis is along the orbital angular momentum at the reference
        epoch. The separation vector from the lighter BH to the heavier BH at
        the reference epoch is along the +ve x-axis. The y-axis completes the
        right-handed triad. The reference epoch is set using f_ref.
        """

        if dfM is not None:
            raise ValueError('Expected dfM to be None for a Time domain model')
        if freqsM is not None:
            raise ValueError('Expected freqsM to be None for a Time domain'
                ' model')

        if mode_list is None:
            mode_list = self.mode_list
        if ellMax is not None:
            if ellMax > np.max(np.array(self.mode_list).T[0]):
                raise ValueError('ellMax is greater than max allowed ell.')
            include_modes = np.array(self.mode_list).T[0] <= ellMax
            mode_list = [self.mode_list[idx]
                    for idx in range(len(self.mode_list))
                    if include_modes[idx]]

        if par_dict is not None:
            raise ValueError('par_dict should be None for this model')

        # always evaluate the (2,2) mode, the other modes neeed this
        # for transformation from coorbital to inertial frame

        # At this stage the phase of the (2,2) mode is the residual after
        # removing the TaylorT3 part (see. Eq.44 of arxiv.1812.07865)
        h_22 = self._eval_sur(x, tuple([2, 2]))

        # Get the TaylorT3 part and add to get the actual phase
        self._set_TaylorT3_factor()
        h_22[0]['phase'] += self._TaylorT3_phase_22(x)

        h_coorb = {k: self._eval_sur(x, k) for k in mode_list \
                        if k != tuple([2,2])}

        return self._coorbital_to_inertial_frame(h_coorb, h_22, \
            mode_list, dtM, timesM, fM_low, fM_ref, do_not_align)

class AlignedSpinCoOrbitalFrameSurrogateTidal(AlignedSpinCoOrbitalFrameSurrogate):
    """
    A surrogate for coorbital frame multimodal waveforms, where each waveform
    data piece has its own surrogate.

    The waveform data pieces are:
    Amplitude and phase of the (2,2) mode.
    Real and imaginary parts of coorbital frame waveform for other modes.

    This generates tidal inspiral waveforms by taking the surrogate output tuned
    to BBH results and incorporates the PN tidal corrections according to the
    tidal splicing method

    NOTE: This returns the waveform only during the inspiral portion of the
    binary's evolution, where the PN expansion is still valid; additional work
    will need to be done in order to complete the merger/ringdown portion of the
    waveform

    NOTE: The waveform is output with the time set so that t=0 corresponds to
    the peak of the waveform for the BBH waveform from the underlying surrogate,
    and NOT the peak of the tidally spliced waveform
    """

    def _coorbital_to_inertial_frame(self, h_coorb, h_22, mode_list, dtM,
        timesM, fM_low, fM_ref, do_not_align, x):
        """ Transforms a dict from Coorbital frame to inertial frame.

            The surrogate data is sparsely sampled, so upsamples to time
            step dtM if given. This is done in the coorbital frame since
            the waveform is slowly varying in that frame.

            If fM_low must be specified. The option of fM_low == 0 has been
            turned off for this model because of its excessive computational
            cost to evaluate

            if do_not_align = False:
                Aligns the 22 mode phase to be 0 at fM_ref. This means
                that at this reference frequency, the heavier BH is roughly on
                the +ve x axis and the lighter BH is on the -ve x axis.
            do_not_align should be True only when converting from pySurrogate
            format to gwsurrogate format as we may want to do some checks that
            the waveform has not been modified
        """

        Amp_22 = h_22[0]['amp']
        phi_22 = h_22[0]['phase']
        domain = np.copy(self.domain)

        # Get omega22_sparse, the angular frequency of the 22 mode, from the
        # sparse surrogate domain.
        # Use np.gradient
        omega22_sparse = np.gradient(phi_22, domain)

        # t=0 is at the waveform peak for the surrogate
        peak22Idx = np.argmin(np.abs(domain))
        omega22_peak = omega22_sparse[peak22Idx]
        # We ignore the part after the peak.  This way we avoid the noisy part
        # at late times, which can randomly be at frequency = fM_low.
        omega22_sparse = omega22_sparse[domain <= domain[peak22Idx]]

        # Get initIdx such that the initial (2, 2) mode frequency ~ fM_low.
        # We will make this more precise below.
        if fM_low != 0:
            omega_low = 2*np.pi*fM_low
            if omega_low < omega22_sparse[0]:
                raise ValueError('f_low is lower than the minimum allowed'
                    ' frequency')
            if omega_low > omega22_peak:
                raise ValueError('f_low is higher than the peak frequency')

            # Choose 5 indices less, to ensure omega_low is included
            initIdx = self._search_omega(omega22_sparse, omega_low) - 5
            # But if initIdx < 0, we are at the start of the surrogate data
            # so just choose 0
            if initIdx < 0:
                initIdx = 0
        else:
            raise ValueError("The option of setting 'fM_low' to 0 is turned off"
                    " for this model; must specifiy a non-zero 'fM_low'")
            ## If fM_low is 0, we use the entire waveform where frequency is
            ## monotonic, uncomment if want to allow this option
            #freq_orbital = np.gradient(phi_22[:peak22Idx], domain[:peak22Idx])
            #if np.min(np.diff(freq_orbital))<=0:
            #  initIdx = len(freq_orbital)-np.argmin((np.diff(freq_orbital)>np.zeros(len(freq_orbital)-1))[::-1])-1
            #else:
            #  initIdx = 0

        Amp_22 = Amp_22[initIdx:peak22Idx]
        phi_22 = phi_22[initIdx:peak22Idx]
        domain = domain[initIdx:peak22Idx]
        v_domain = np.power(np.abs(np.gradient(phi_22, domain))/2,1./3.)
        if(np.min(np.diff(v_domain))<0):
            raise ValueError('frequency is not monotonic over the entire'
                ' considered here')

        if timesM is not None:
            # This check is performed after the tidal terms computed
            #if timesM[-1] > domain[-1]:
            #    raise Exception("'times' includes times larger than the"
            #        " maximum time value in domain.")
            if timesM[0] < domain[0]:
                raise Exception("'times' starts before start of domain. Try"
                    " increasing initial value of times or reducing f_low.")

        # For tidal splicing, always want to interpolate first to a dense domain
        # in order to compute an accurate orbital frequency for the PN equations
        # then interpolated to the desired times afterwards

        if dtM is None and timesM is None:
            raise ValueError("For this model, must specify either the 'dtM' or"
                " 'timesM' option")
        else:
            ## Interpolate onto uniform domain
            ## WARNING -- if the the time points are not sampled densely enough
            ## here, there is a potential for error due to inaccurate orbital
            ## freq being used for the PN tidal equations
            if dtM is not None:
                t0 = domain[0]
                tf = domain[-1]
                num_times = int(np.ceil((tf - t0)/dtM));
                timesM_tmp = t0 + dtM*np.arange(num_times)
            else:
                # Because the spliced waveform is shifted so the final time
                # is the peak of the final waveform, we must ensure the check
                # here is performed similarly
                if timesM[0] < (domain[0]-domain[-1]) or timesM[-1] > 0:
                    raise Exception('Trying to evaluate at times outside the'
                        ' domain.')
                min_dt = np.min(np.diff(timesM))
                t0 = domain[0] #timesM[0] - min_dt
                tf = domain[-1]
                num_times = int(np.ceil((tf - t0)/min_dt));
                timesM_tmp = t0 + min_dt*np.arange(num_times)

            Amp_22 = _splinterp_Cwrapper(timesM_tmp, domain, Amp_22)
            phi_22 = _splinterp_Cwrapper(timesM_tmp, domain, phi_22)

            # now recompute omega22 with the dense data, but retain only data
            # upto the peak to avoid the noisy part
            omega22 = np.gradient(phi_22, timesM_tmp)

            #omega22 = omega22[timesM_tmp <= timesM_tmp[np.argmax(Amp_22)]]

            # Truncate data so that only freqs above omega_low are retained
            # If timesM are already given, we don't need to truncate data
            if dtM is not None:
                if fM_low != 0:
                    startIdx = max(np.argmin(np.abs(omega22 - omega_low)) - 4,0)
                else:
                    raise ValueError("The option of setting 'fM_low' to 0"
                            " is turned off for this model; must specifiy a"
                            " non-zero 'fM_low'")
                    ## If fM_low is 0, we use the entire waveform that is monotonic
                    #startIdx = 0
                    #if np.min(np.diff(omega22))<=0:
                    #  startIdx = len(omega22)-np.argmin((np.diff(omega22)>np.zeros(len(omega22)-1))[::-1])-1
                    ## Because the splicing changes the frequencies slightly, to
                    ## ensure we have wiggle room for interpolation later, buffer
                    ## the altered initial frequency of the spliced waveform so
                    ## it is not less than the initial frequency of v_domain
                    #gap = int((domain[5]-domain[0])/(timesM_tmp[1]-timesM_tmp[0]))
                    #if startIdx<gap:
                    #  startIdx=gap

                Amp_22 = Amp_22[startIdx:]
                phi_22 = phi_22[startIdx:]
                omega22 = omega22[startIdx:]
                timesM_tmp = timesM_tmp[startIdx:]

        freq_orbital = np.abs(omega22)/2
        v = np.power(freq_orbital,1./3.)

        # Setup all of the tidal parameters
        # Use universal relations to compute parameters beyond the quad love num
        # NOTE: omega2AB and omega3AB are stored as M*omega{2,3}{A,B}, to use the
        #   dimensionless value set the total mass of the system and that the
        #   universal relations return M{A,B}*omega{2,3}{A,B}
        # Aqm is the dimensionless quadrupole moment, however for splicing the 2PN BBH
        #   (v^4) term, the qm of a BBH must be subtracted off (Aqm_BH = 1; see
        #   arXiv:gr-qc/9709032 just below eqn 8), which will be done in the tidal
        #   function itself
        # If the NS is spinning, the effective driving frequency that the NS sees,
        #   from its own reference frame, is shifted according to the NS spin by the
        #   dimensionless rotation = M omega_spin =  (M / m_NS) * chi_NS / \bar{I}
        # WARNING: This effect has been turned off (omega_spin = 0) for NS anti-
        #   aligned spins, b/c the resonance peak is shifted to early enough in the
        #   inspiral that the approximation being used to model it might be breaking
        #   down by the end of the late inspiral (only supposed to be good up until
        #   shortly after resonance)
        qqq = x[0]; chiAz = x[1]; chiBz = x[2]; lambda2A = x[3]; lambda2B = x[4]
        XA = qqq/(1.+qqq); XB = 1.-XA
        omega2A = lambda3A = omega3A = AqmA = 0.
        omega2B = lambda3B = omega3B = AqmB = 0.
        omegaSpinA = omegaSpinB = 0.
        ell2Adyn = ell2Adiss = ell2Bdyn = ell2Bdiss = np.zeros(len(timesM_tmp))
        ell3Adyn = ell3Bdyn = np.zeros(len(timesM_tmp))
        if(lambda2A>0):
            IbarA     = UniversalRelationLambda2ToI(lambda2A)
            omegaSpinA = max(chiAz,0) / IbarA / XA
            omega2A   = UniversalRelationLambda2ToOmega2(lambda2A)/XA
            lambda3A  = UniversalRelationLambda2ToLambda3(lambda2A)
            omega3A   = UniversalRelationLambda3ToOmega3(lambda3A)/XA
            AqmA      = UniversalRelationLambda2ToAqm(lambda2A)
            ell2Adyn  = EffectiveDeformabilityFromDynamicalTides \
                        (np.abs(freq_orbital-omegaSpinA),omega2A,2,qqq)
            ell3Adyn  = EffectiveDeformabilityFromDynamicalTides \
                        (np.abs(freq_orbital-omegaSpinA),omega3A,3,qqq)
        if(lambda2B>0):
            IbarB     = UniversalRelationLambda2ToI(lambda2B)
            omegaSpinB = max(chiBz,0) / IbarB / XB
            omega2B   = UniversalRelationLambda2ToOmega2(lambda2B)/XB
            lambda3B  = UniversalRelationLambda2ToLambda3(lambda2B)
            omega3B   = UniversalRelationLambda3ToOmega3(lambda3B)/XB
            AqmB      = UniversalRelationLambda2ToAqm(lambda2B)
            ell2Bdyn  = EffectiveDeformabilityFromDynamicalTides \
                        (np.abs(freq_orbital-omegaSpinB),omega2B,2,qqq)
            ell3Bdyn  = EffectiveDeformabilityFromDynamicalTides \
                        (np.abs(freq_orbital-omegaSpinB),omega3B,3,qqq)

        dt_tid, dp_tid = PNT2Tidal(v, qqq, lambda2A*ell2Adyn, \
                lambda3A*ell3Adyn, AqmA, chiAz, lambda2B*ell2Bdyn, \
                lambda3B*ell3Bdyn, AqmB, chiBz, order=5)

        timesM_tmp += dt_tid - dt_tid[0]

        # Limit the waveform to the last time in the array that is increasing
        find = np.argmin(np.diff(timesM_tmp)>0)
        if(find == 0):
            find = len(timesM_tmp)

        timesM_tmp = timesM_tmp[:find]

        # There is a small region of parameter space where the interpolation
        # behaves poorly at very late times due to oddly shaped steps, so we
        # need to check the final handful of steps for that and truncate as
        # needed to avoid interpolation failures
        numcheck = 500
        factorLimit = 2.
        tdiff = np.diff(timesM_tmp[-numcheck-1:])
        for i in range(len(tdiff)-1):
          if ((tdiff[i]>tdiff[i+1]*factorLimit) or (tdiff[i]<tdiff[i+1]/factorLimit)):
            find = len(timesM_tmp)-numcheck+i-1
            timesM_tmp = timesM_tmp[:find]
            break

        timesM_tmp -= timesM_tmp[-1]
        phi_22 = phi_22[:find] + 2.*(dp_tid[:find] - dp_tid[0])

        # Reinterpolate to the final time grid
        if dtM is not None:
            t0 = timesM_tmp[0]
            tf = timesM_tmp[-1]
            num_times = int(np.ceil((tf - t0)/dtM));
            timesM = t0 + dtM*np.arange(num_times)
            timesM -= timesM[-1] # Ensure peak amplitude at t=0
        else:
            if timesM[-1] > timesM_tmp[-1]:
                raise Exception("'times' includes times larger than the"
                    " maximum time value in domain after splicing. (Remember"
                    " that tidal effects cause the binary to merger earlier)")
            if timesM[0] < timesM_tmp[0]:
                raise Exception("'times' includes times smaller than the"
                    " initial time value in domain after splicing. (Remember"
                    " that tidal effects cause the binary to merger earlier)")

        # Find the 'v' corresponding to the final time array, then perform the
        # interpolation in the 'v' domain as that is where most of the PN
        # quantities are defined
        v_uniform = _splinterp_Cwrapper(timesM, timesM_tmp, v[:find])

        Amp_22 = _splinterp_Cwrapper(v_uniform, v[:find], Amp_22[:find])
        phi_22 = _splinterp_Cwrapper(v_uniform, v[:find], phi_22)
        freq_orbital = np.power(v_uniform,3.)

        # Dynamical Tidal deformability stuff on final array for strain
        # amplitudes
        ell2Adyn = ell2Adiss = ell2Bdyn = ell2Bdiss = np.zeros(len(timesM))
        if(lambda2A>0):
          ell2Adyn  = EffectiveDeformabilityFromDynamicalTides \
                      (np.abs(freq_orbital-omegaSpinA),omega2A,2,qqq)
          ell2Adiss = EffectiveDissipativeDynamicalTides \
                      (np.abs(freq_orbital-omegaSpinA),ell2Adyn,omega2A,XA)
        if(lambda2B>0):
          ell2Bdyn  = EffectiveDeformabilityFromDynamicalTides \
                      (np.abs(freq_orbital-omegaSpinB),omega2B,2,qqq)
          ell2Bdiss = EffectiveDissipativeDynamicalTides \
                      (np.abs(freq_orbital-omegaSpinB),ell2Bdyn,omega2B,XB)

        # Get reference index where waveform needs to be aligned.
        if (abs(fM_ref-fM_low) < 1e-13) and (dtM is not None):
            # This means that the data is already truncated at fM_low,
            # so we just need the first index for fM_ref=fM_low
            refIdx = 0
        else:
            omega_ref = 2*np.pi*fM_ref
            if omega_ref > omega22_peak:
                raise ValueError('f_ref is higher than the peak frequency')

            refIdx = np.argmin(np.abs(2.*freq_orbital - omega_ref))


        # do_not_align should be True only when converting from pySurrogate
        # format to gwsurrogate format as we may want to do some checks that
        # the waveform has not been modified
        if not do_not_align:
            # Set orbital phase to 0 refIdx. Note that the Coorbital
            # frame data is not affected by this constant phase shift.

            # The orbital phase is obtained as phi_22/2, so this leaves a pi
            # ambiguity.  But the surrogate data is already aligned such that
            # the heavier BH is on the +ve x-axis at t=-1000M. See Sec.VI.A.4
            # of arxiv:1812.07865, the resolves the pi ambiguity. This means
            # that the after the realignment, the orbital phase at reference
            # frequency is 0.
            phi_22 += -phi_22[refIdx]


        h_dict = {}
        for mode in mode_list:
            if mode == tuple([2, 2]):
                h_dict[mode] = (Amp_22+StrainTidalEnhancementFactor(2,2, \
                      qqq,(lambda2A*ell2Adiss),(lambda2B*ell2Bdiss),v_uniform)) \
                      * np.exp(-1j*phi_22)
            else:
                l,m = mode
                h_coorb_lm = 0
                if 're' in h_coorb[mode][0].keys():
                    h_coorb_lm += h_coorb[mode][0]['re'] + 1j * 0
                if 'im' in h_coorb[mode][0].keys():
                    h_coorb_lm += 1j*h_coorb[mode][0]['im']

                h_coorb_lm = h_coorb_lm[initIdx:peak22Idx]

                h_coorb_lm = _splinterp_Cwrapper(v_uniform,v_domain,h_coorb_lm)

                h_coorb_lm_amp = np.abs(h_coorb_lm)
                h_coorb_lm_phase = np.unwrap(np.angle(h_coorb_lm))
                h_coorb_lm_tid = StrainTidalEnhancementFactor(l,m,qqq, \
                        (lambda2A*ell2Adiss),(lambda2B*ell2Bdiss),v_uniform)
                h_dict[mode] = (h_coorb_lm_amp + h_coorb_lm_tid) \
                        * np.exp((-1j*m/2.)*phi_22+1j*h_coorb_lm_phase)

        return timesM, h_dict, None     # None is for dynamics

    def __call__(self, x, fM_low=None, fM_ref=None, dtM=None,
        timesM=None, dfM=None, freqsM=None, mode_list=None, ellMax=None,
        precessing_opts=None, tidal_opts=None, par_dict=None,
        do_not_align=False):
        """
    Return dimensionless surrogate modes.
    Arguments:
    x :             The intrinsic parameters EXCLUDING total Mass (see
                    self.param_space)

    fM_low :        Initial frequency of (2,2) mode in units of cycles/M.
                    If 0, will use the entire data of the surrogate.
                    Default None.

    fM_ref:         Frequency used to set the reference epoch at which
                    the reference frame is defined and the spins are specified.
                    See below for definition of the reference frame.
                    Default: None.

                    For time domain models, f_ref is used to determine a t_ref,
                    such that the frequency of the (2, 2) mode equals f_ref at
                    t=t_ref.

    dtM :           Uniform time step to use, in units of M. If None, the
                    returned time array will be the array used in the
                    construction of the surrogate, which can be nonuniformly
                    sampled.
                    Default None.

    timesM:         Time samples to evaluate the waveform at. Use either dtM or
                    timesM, not both.

    dfM :           This should always be None as for now we are assuming
                    a time domain model.

    freqsM:         Frequency samples to evaluate the waveform at. Use either
                    dfM or freqsM, not both.

    ellMax:         Maximum ell index for modes to include. All available m
                    indicies for each ell will be included automatically.
                    Default: None, in which case all available modes wll be
                    included.

    mode_list :     A list of (ell, m) modes to be evaluated.
                    Default None, which evaluates all avilable modes.
                    Will deduce the m<0 modes from m>0 modes.

    par_dict:       This should always be None for this model.

    do_not_align:   Ignore fM_ref and do not align the waveform. This should be
                    True only when converting from pySurrogate format to
                    gwsurrogate format as we may want to do some checks that
                    the waveform has not been modified.

    Returns
    timesM, h, dynamics:
        timesM : time array in units of M.
        h : A dictionary of waveform modes sampled at timesM with
            (ell, m) keys.
        dynamics: None, since this is a nonprecessing model.


    IMPORTANT NOTES:
    ===============

    The reference frame (or inertial frame) is defined as follows:
        The +ve z-axis is along the orbital angular momentum at the reference
        epoch. The separation vector from the lighter BH to the heavier BH at
        the reference epoch is along the +ve x-axis. The y-axis completes the
        right-handed triad. The reference epoch is set using f_ref.
        """

        if par_dict is not None:
            raise ValueError('Expected par_dict to be None.')
        if dfM is not None:
            raise ValueError('Expected dfM to be None for a Time domain model')
        if freqsM is not None:
            raise ValueError('Expected freqsM to be None for a Time domain'
                ' model')

        if mode_list is None:
            mode_list = self.mode_list
        if ellMax is not None:
            if ellMax > np.max(np.array(self.mode_list).T[0]):
                raise ValueError('ellMax is greater than max allowed ell.')
            include_modes = np.array(self.mode_list).T[0] <= ellMax
            mode_list = [self.mode_list[idx]
                    for idx in range(len(self.mode_list))
                    if include_modes[idx]]

        # The last to parameters are the tidal parameters and are not a part of
        # the base surrogate model
        x_sur = x[:-2]

        # always evaluate the (2,2) mode, the other modes neeed this
        # for transformation from coorbital to inertial frame

        # At this stage the phase of the (2,2) mode is the residual after
        # removing the TaylorT3 part (see. Eq.44 of arxiv.1812.07865)
        h_22 = self._eval_sur(x_sur, tuple([2, 2]))

        # Get the TaylorT3 part and add to get the actual phase
        self._set_TaylorT3_factor()
        h_22[0]['phase'] += self._TaylorT3_phase_22(x_sur)

        h_coorb = {k: self._eval_sur(x_sur, k) for k in mode_list \
                        if k != tuple([2,2])}

        return self._coorbital_to_inertial_frame(h_coorb, h_22, \
            mode_list, dtM, timesM, fM_low, fM_ref, do_not_align, x)



class SpEC_nonspinning_q10_surrogate(MultiModalSurrogate):
    """A special class for the SpEC nonspinning surrogate"""

    skip_2_0_mode = True

    def __call__(self, x, theta=None, phi=None, modes=None,
                 fake_neg_modes=True):
        """
        Return surrogate evaluation.
        Arguments:
            x : The intrinsic parameters (see self.param_space)
            theta/phi : polar and azimuthal angles of the direction of
                        gravitational wave emission. If given, sums up modes
                        and returns h_plus and h_cross (default returns modes)
            modes : A list of (ell, m) modes to be evaluated (default: all)
            fake_neg_modes: Deduce (ell, -m) modes from (ell, m) modes for m>0.
        Returns h:
            h : If theta and phi are None, h is a dictionary of waveform modes
                sampled at self.domain with (ell, m) keys.
                If theta and phi are given, h = h_plus - i * h_cross is a
                complex array given by the sum of the modes.
        """
        if modes is None:
            modes = self.modes

        if self.skip_2_0_mode:
            # removed by 2to3 tool
            #modes = filter(lambda mode: mode != (2, 0), modes)
            modes = [mode for mode in modes if mode != (2, 0)]

        h_modes = super(SpEC_nonspinning_q10_surrogate, self).__call__(
                x, modes=modes)

        if fake_neg_modes:
            new_modes = {}
            for (ell, m), h in h_modes.items(): # inefficient in py2
                if m > 0:
                    new_modes[(ell, -m)] = np.power(-1, ell) * h.conjugate()
            for k, v in new_modes.items(): # inefficient in py2
                h_modes[k] = v

        if theta is not None:
            return _mode_sum(h_modes, theta, phi)

        return h_modes
