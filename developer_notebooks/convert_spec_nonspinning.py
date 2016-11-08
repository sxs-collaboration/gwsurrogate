import h5py
import gwsurrogate.new.surrogate as surrogate
import gwsurrogate.new.nodeFunction as nodeFunction
import numpy as np

def chars_to_string(chars):
    return "".join(chr(cc) for cc in chars)

def make_node(g, i, name, phase=False):
    if phase:
        fname = chars_to_string(g['fit_type_phase'].value)
        coefs = g['fitparams_phase'].value[i]
    else:
        fname = chars_to_string(g['fit_type_amp'].value)
        coefs = g['fitparams_amp'].value[i]
    node = nodeFunction.MappedPolyFit1D_q10_q_to_nu(function_name=fname, coefs=coefs)
    nf = nodeFunction.NodeFunction(name=name, node_function=node)
    return nf

def get_mode_surrogate_data(g, name):
    B = g['B'].value.T
    B_phase = g['B_phase'].value.T
    nodes = []
    for i in range(len(g['fitparams_amp'].value)):
        nodes.append(make_node(g, i, '%s_amp_node_%s'%(name, i)))
    phase_nodes = []
    for i in range(len(g['fitparams_phase'].value)):
        phase_nodes.append(make_node(g, i, '%s_phase_node_%s'%(name, i), True))
    return {'amp': (B, nodes), 'phase': (B_phase, phase_nodes)}


f = h5py.File('../surrogates/SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0.h5')

lm_modes = [(int(k[1]), int(k[4])) for k in f.keys()]
mode_data = {}
for (ell, m) in lm_modes:
    print 'loading (%s, %s)...'%(ell, m)
    g = f['l%s_m%s'%(ell, m)]
    name = 'mode_%s_%s'%(ell, m)
    mode_data[(ell, m)] = get_mode_surrogate_data(g, name)

t = np.arange(-2750.0, 100.00001, 0.1)

param_dim_q = surrogate.ParamDim('q', 1, 10)
param_space = surrogate.ParamSpace('Nonspinning_q_1_10', [param_dim_q])

sur = surrogate.MultiModalSurrogate(
        name='SpEC_q1_10_nonspinning_surrogate',
        domain=t,
        param_space=param_space,
        mode_data=mode_data,
        mode_type='amp_phase',
        modes=lm_modes,
    )

print 'done making surrogate, testing...'
res = sur(1.1)
res2 = sur(9.9)
print 'saving...'
sur.save('new_SpEC_sur.h5')

print 'loading...'
sur2 = surrogate.MultiModalSurrogate()
sur2.load('new_SpEC_sur.h5')
print 'verifying...'
okay=True
new_res = sur2(1.1)
new_res2 = sur2(9.9)
for old, new in [(res, new_res), (res2, new_res2)]:
    for mode in sur.modes:
        if np.max(abs(old[mode] - new[mode])) > 0.:
            print 'Got a difference in the %s mode!'%(mode)
            okay = False

if okay:
    print 'Great, results match!'
