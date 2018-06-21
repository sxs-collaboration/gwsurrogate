""" Gravitational Wave Surrogate classes for text and hdf5 files"""

from __future__ import division  # for py2

__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@astro.cornell.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Jonathan Blackman, Scott Field, Chad Galley, Vijay Varma"

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
from gwsurrogate.gwtools.harmonics import sYlm as _sYlm

if __package__ is "" or "None": # py2 and py3 compatible
  print("setting __package__ to gwsurrogate.new so relative imports work")
  __package__="gwsurrogate.new"

# assumes unique global names
from .saveH5Object import SimpleH5Object
from .saveH5Object import H5ObjectList
from .saveH5Object import H5ObjectDict
from .nodeFunction import NodeFunction
from .spline_evaluation import TensorSplineGrid, fast_complex_tensor_spline_eval

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
                            tolerance, name, min_val, max_val))

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

    def __init__(self, name=None, domain=None, param_space=None,
            coorb_mode_data={(2, 2): {}}
            ):
        """
        name:               A descriptive name for this surrogate.

        domain:             A 1d array of the monotonically increasing time
                            values.

        param_space:        A ParamSpace for this surrogate.

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

        super(AlignedSpinCoOrbitalFrameSurrogate, self).__init__(name,
                domain, param_space, {}, many_function_components,
                self.mode_type)

        self._h5_data_keys.append('mode_list')
        self._h5_data_keys.append('mode_type')


    def _coorbital_to_inertial_frame(self, h_coorb, h_22, mode_list, dtM,
        timesM, fM_low, fM_ref, do_not_align):
        """ Transforms a dict from Coorbital frame to inertial frame.

            The surrogate data is sparsely sampled, so upsamples to time
            step dtM if given. This is done in the coorbital frame since
            the waveform is slowly varying in that frame.

            If fM_low is given, only part of the waveform where frequency of
            the (2, 2) mode is greater than fM_low is retained.

            if do_not_align = False:
                Aligns the 22 mode phase to be zero at fM_ref. This means that
                at this reference frequency, the larger BH is roughly on the
                +ve x axis and the smaller BH is on the -ve x axis.
            do_not_align should be True only when converting from pySurrogate
            format to gwsurrogate format as we may want to do some checks that
            the waveform has not been modified
        """

        Amp_22 = h_22[0]['amp']
        phi_22 = h_22[0]['phase']
        domain = np.copy(self.domain)

        # get omega22, the angular frequency of the 22 mode, but truncate the
        # late time ( > 50 from peak). This way we avoid the noisy part at
        # late times, which can randomly be at frequency = fM_low.
        omega22 = np.gradient(phi_22)/np.gradient(domain)
        peak22Idx = np.argmax(Amp_22)
        omega22 = omega22[domain <= domain[peak22Idx]+50]

        # Truncate waveform such that the initial (2, 2) mode frequency = fM_low
        if fM_low is not None:
            omega_low = 2*np.pi*fM_low
            if omega_low < omega22[0]:
                raise ValueError('f_low is lower than the minimum allowed'
                    ' frequency')
            startIdx = np.argmin(np.abs(omega22 - omega_low))
            if domain[startIdx] > domain[peak22Idx] + 10:
                raise Exception('The time that matches f_low is after the peak,'
                    ' something must be wrong.')

            Amp_22 = Amp_22[startIdx:]
            phi_22 = phi_22[startIdx:]
            domain = domain[startIdx:]
            omega22 = omega22[startIdx:]

            if timesM is not None:
                if timesM[0] < domain[0]:
                    raise Exception("'times' starts before start of domain. Try"
                        " increasing initial value of times or reducing f_low.")
                if timesM[-1] > domain[-1]:
                    raise Exception("'times' includes times larger than the"
                        " maximum time value in domain.")

        return_times = True
        if dtM is None and timesM is None:
            timesM = domain
            do_interp = False
        else:
            do_interp = True
            if dtM is not None:
                # Interpolate onto uniform domain if needed
                timesM = np.arange(domain[0], domain[-1], dtM)
            else:
                return_times = False
                if timesM[0] < domain[0] or timesM[-1] > domain[-1]:
                    raise Exception('Trying to evaluate at times outside the'
                        ' domain.')
            Amp_22 = _splinterp(timesM, domain, Amp_22, ext='raise')
            phi_22 = _splinterp(timesM, domain, phi_22, ext='raise')


        # Get reference index where waveform needs to be aligned. If fM_ref
        # is not given, we pick the first index
        if fM_ref is not None:
            refIdx = np.argmin(np.abs(omega22 - 2*np.pi*fM_ref))
            if timesM[refIdx] > timesM[np.argmax(Amp_22)] + 10:
                raise Exception('The time that matches f_ref is after the peak,'
                    ' something must be wrong.')
        else:
            refIdx = 0

        # do_not_align should be True only when converting from pySurrogate
        # format to gwsurrogate format as we may want to do some checks that the
        # waveform has not been modified
        if not do_not_align:
            # Align phase at refIdx. Note that the Coorbital frame data is not
            # affected by this constant phase shift.
            phi_22 -= phi_22[refIdx]

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

                if fM_low is not None:
                    h_coorb_lm = h_coorb_lm[startIdx:]
                if do_interp:
                    h_coorb_lm = _splinterp(timesM, domain, h_coorb_lm, \
                        ext='raise')
                h_dict[mode] = h_coorb_lm * np.exp(-1j*m*phi_22/2.)

        if return_times:
            return timesM, h_dict
        else:
            return h_dict


    def __call__(self, x, fM_low=None, fM_ref=None, dtM=None, timesM=None,
        dfM=None, freqsM=None, mode_list=None, par_dict=None,
        do_not_align=False):
        """
    Return dimensionless surrogate modes.
    Arguments:
    x :             The intrinsic parameters EXCLUDING total Mass (see
                    self.param_space)

    fM_low :        Initial frequency of (2,2) mode in units of cycles/M.
                    If None, will use the entire data of the surrogate.
                    Default None.

    fM_ref:         Reference frequency used to set the reference epoch at which
                    the frame is aligned and the spins are specified. The frame
                    is aligned at the reference epoch as follows:
                        The orbital angular momentum points towards the
                            +ve z-axis.
                        The separation vector from the smaller BH to the larger
                            BH points towards the +ve x-axis.
                    For time domain models, this is used to detemine a t_ref,
                    such that the frequency of the (2, 2) mode equals fM_ref
                    at t=t_ref.
                    Default: If fM_low is given, fM_ref = fM_low. If fM_low is
                    None, fM_ref is set to the initial frequency (the first
                    index).

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

    mode_list :     A list of (ell, m) modes to be evaluated.
                    Default None, which evaluates all avilable modes.
                    Will deduce the m<0 modes from m>0 modes.

    par_dict:       This should always be None for this model.

    do_not_align:   Ignore fM_ref and do not align the waveform. This should be
                    True only when converting from pySurrogate format to
                    gwsurrogate format as we may want to do some checks that
                    the waveform has not been modified.

    Returns 
    h: If timesM is given.
    timesM, h: If timesM is None.
        timesM : time array in units of M.
        h : A dictionary of waveform modes sampled at times=t with
            (ell, m) keys.
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

        # always evaluate the (2,2) mode, the other modes neeed this
        # for transformation from coorbital to inertial frame
        h_22 = self._eval_sur(x, tuple([2, 2]))

        h_coorb = {k: self._eval_sur(x, k) for k in mode_list \
                        if k != tuple([2,2])}

        return self._coorbital_to_inertial_frame(h_coorb, h_22, \
            mode_list, dtM, timesM, fM_low, fM_ref, do_not_align)


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
