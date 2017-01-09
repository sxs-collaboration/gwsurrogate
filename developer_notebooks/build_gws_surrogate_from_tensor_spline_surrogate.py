import gwsurrogate as gws
import gwsurrogate.new.surrogate as gwsnew
from gwsurrogate import surrogateIO
from gwsurrogate.new.spline_evaluation import cubic_spline_breaksToknots
import numpy as np

ts_filename = 'new_NR_4d2s_TD_5_4_7_4_6.h5'
out_filename = 'test_surrogate.h5'
N_MODES = 12

def save_gws_surrogate(filename, lm_modes, ei, v, cre, cim):
    writeh5 = surrogateIO.H5Surrogate(filename, 'w')

    for i in range(N_MODES):
        ell, m = lm_modes[i]
        smd = {}
        smd['surrogate_mode_type'] = 'waveform_basis'

        # This doesn't make much sense for parameter space dimensions > 1
        # Should have something like smd['parameterization'] = '5d2s'
        smd['parameterization'] = 'q_to_q'

        smd['affine_map'] = 'none' # Not sure how this generalizes to higher d
        smd['t_units'] = 'TOverMtot'
        smd['B'] = ei[i]
        smd['V'] = v[i]
        smd['fit_min'] = np.array([1.0, 0.0, 0.0, 0.0, -0.8])
        smd['fit_max'] = np.array([2.0, 0.8, np.pi, 2*np.pi, 0.8])
        smd['times'] = sur.domain

        # This is ugly. Suggest having an object for each parameter dimension
        # storing min_val, max_val, spline knots, and anything else relevant
        # to that dimension. These should be savable and loadable as h5 groups.
        # This was the smallest change I could think of to make it work.
        bvecs = sur.ts_grid.breakpoint_vecs
        knots = [cubic_spline_breaksToknots(b) for b in bvecs]
        smd['n_spline_knots'] = np.array([len(k) for k in knots])
        all_knots = []
        for k in knots:
            all_knots += list(k)
        smd['spline_knots'] = np.array(all_knots)
        
        smd['fitparams_amp'] = cre[i]
        smd['fitparams_phase'] = cim[i]
        smd['fit_type_phase'] = 'fast_spline_imag'
        smd['fit_type_amp'] = 'fast_spline_real'
        save_subdir = 'l%s_m%s'%(ell, m)
        print 'saving mode %s'%(save_subdir)
        writeh5.write_h5(smd, subdir=save_subdir, closeQ = (i==N_MODES-1))

print 'loading TS surrogate...'
sur = gwsnew.FastTensorSplineSurrogate()
sur.load(ts_filename)

# These are useful to have in gws surrogates. Not sure if mandatory.
v = [np.load('vandermond/mode%s.npy'%(i)) for i in range(12)]

lm_modes = []
for ell in range(2, 4):
    for m in range(-ell, ell+1):
        lm_modes.append((ell, m))

print 'Saving gws surrogate...'
save_gws_surrogate(out_filename, lm_modes, sur.ei, v, sur.cre, sur.cim)

print 'Loading gws surrogate...'
sur_new = gws.EvaluateSurrogate(out_filename, use_orbital_plane_symmetry=False)

print 'Testing...'
x = np.array([1.2, 0.3, 0.4, 0.5, -0.2])
h_ts = sur(x)

lm_modes, t, hre, him = sur_new(x, mode_sum=False, fake_neg_modes=False)
h_gws = (hre + 1.j*him).T

print 'Evaluation was successfull! Checking errors...'

max_err = 0.
for mode_gws, (ell, m) in zip(h_gws, lm_modes):
    mode_ts = h_ts[ell, m]
    err = np.max(abs(mode_ts - mode_gws))
    max_err = max(err, max_err)
    print 'Max (%s, %s) mode error: %s'%(ell, m, err)

print 'Max error: %s'%(max_err)
if max_err > 1.e-10:
    print 'Surrogates disagree :('
else:
    print 'Good agreement!'
