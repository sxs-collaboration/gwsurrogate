import gwsurrogate.new.surrogate as surrogate
import numpy as np

n = 7
data_dir='.'
save_prefix = 'NR_4d2s_TD_TSgrid'
lmax = 3

######
filename = '%s%s.h5'%(save_prefix, n)
d = '%s/NR_v4_TD_%s'%(data_dir, n)
dim=(100, n+2, n+2, n+2, n+2, n+2)


lm_modes = [(ell, m) for ell in range(2, lmax+1) for m in range(-ell, ell+1)]
n_modes = len(lm_modes) + 1

# Load data
print 'loading...'
t = np.load('%s/NRSurrogate_times.npy'%d)
ei = [np.load('%s/NRSurrogate_EI_%s.npy'%(d, i)) for i in range(12)]
cre = [np.load('%s/NRSurrogate_cre_%s.npy'%(d, i)).reshape(*dim)
       for i in range(12)]
cim = [np.load('%s/NRSurrogate_cim_%s.npy'%(d, i)).reshape(*dim)
       for i in range(12)]
mode_data = {k: (ei[i], cre[i], cim[i]) for i, k in enumerate(lm_modes)}

# Make parameter space
print 'building...'
q = surrogate.ParamDim('q', 1, 2)
chiA = surrogate.ParamDim('|chiA|', 0, 0.8)
theta = surrogate.ParamDim('theta', 0, np.pi)
phi = surrogate.ParamDim('phi', 0, 2*np.pi)
chiBz = surrogate.ParamDim('chiB_z', -.8, .8)
param_space = surrogate.ParamSpace('4d2s', [q, chiA, theta, phi, chiBz])
knots = []
for xmin, xmax in zip(param_space.min_vals(), param_space.max_vals()):
    knots.append(np.linspace(xmin, xmax, 5))

# Build surrogate
sur = surrogate.FastTensorSplineSurrogate(
    name='%s%s'%(save_prefix, n),
    domain=t,
    param_space=param_space,
    knot_vecs=knots,
    mode_data=mode_data,
    modes=lm_modes,
        )

# Test evaluation
print 'evaluating and saving...'
x = np.array([1.2, 0.2, 0.2, 0.2, 0.2])
h = sur(x)
sur.save(filename)

# Load and validate
print 'validating...'
sur2 = surrogate.FastTensorSplineSurrogate()
sur2.load(filename)
h2 = sur2(x)
okay=True
for key in lm_modes:
    print key, np.max(abs(h[key] - h2[key]))
    if np.max(abs(h[key] - h2[key])) > 0.:
        okay = False

if okay:
    print 'Looks good!'
else:
    raise Exception('Loaded surrogate does not match saved surrogate!')
