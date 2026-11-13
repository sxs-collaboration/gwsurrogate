# NRHybSur3dq8

[![arXiv:1812.07865](https://img.shields.io/badge/arXiv-1812.07865-B31B1B.svg)](https://arxiv.org/abs/1812.07865)
[![Data](https://img.shields.io/badge/data-Zenodo-1682D4.svg)](https://zenodo.org/record/3348115)

A surrogate model for **aligned-spin** (nonprecessing) binary black holes. It was
trained on numerical relativity (NR) waveforms that were hybridized with
post-Newtonian (PN) and effective-one-body (EOB) waveforms at early times, so the
model spans the entire LIGO frequency band for stellar-mass binaries. See
[arXiv:1812.07865](https://arxiv.org/abs/1812.07865) (Varma et al. 2018) for details of the hybridization,
model construction, and accuracy studies.

The CCE-based sibling [NRHybSur3dq8_CCE](NRHybSur3dq8_CCE.md) also captures
gravitational-wave memory, and [NRHybSur3dq8Tidal](NRHybSur3dq8Tidal.md) adds tidal
effects for neutron-star binaries on top of this model.

## Parameter space of validity

| Quantity | Trained / tested range | Soft limit | Hard limit |
|---|---|---|---|
| Mass ratio $q = m_A/m_B \ge 1$ | $q \in [1, 8]$ | $q \approx 8.01$ | $q \approx 10.01$ |
| Aligned spins $\chi_{Az}, \chi_{Bz}$ | $[-0.8, 0.8]$ | $\lvert\chi\rvert \approx 0.801$ | $\lvert\chi\rvert = 1$ |
| Harmonic modes $h_{\ell m}$ | $(2,2), (2,1), (2,0), (3,3), (3,2), (3,1), (3,0), (4,4), (4,3), (4,2), (5,5)$, and the corresponding $m<0$ modes | — | — |

The model is trained within the range shown above, but produces reasonable
waveforms up to the hard limits, where it has been tested against existing NR
waveforms; see [arXiv:1812.07865](https://arxiv.org/abs/1812.07865).

Crossing a **soft limit** prints a warning but still evaluates the surrogate;
crossing a **hard limit** raises an error and blocks evaluation.

Only the $m \ge 0$ modes are modeled; the $m<0$ modes are deduced from the
orbital-plane symmetry of nonprecessing binaries,
$h_{\ell,-m} = (-1)^\ell\, h_{\ell m}^*$.

## Loading the surrogate

The surrogate data only needs to be downloaded and loaded once per script/session.

```python
import gwsurrogate

gwsurrogate.catalog.pull('NRHybSur3dq8')   # one-time download
sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')
```

## Calling the surrogate

The surrogate can be evaluated in two unit systems: dimensionless units (default)
and physical (MKS) units.

### Dimensionless units (default)

This uses geometrized units ($G = c = 1$), where the total mass $M$ sets the sole
physical scale. Masses/times/lengths are in units of $M$, frequencies are in
cycles/$M$, and spins are already dimensionless. `h` is returned in code units,
$rh/M$; `t` is returned in units of $M$.

```python
q = 4                       # mass ratio, mA/mB >= 1 (dimensionless)
chiA = [0, 0, 0.5]          # dimensionless spin of the heavier BH (z-component only)
chiB = [0, 0, -0.7]         # dimensionless spin of the lighter BH (z-component only)
dt = 0.1                    # timestep size, in units of M
f_low = 5e-3                # initial frequency of the (2,2) mode, in cycles/M

# h is a dict of spin-weighted spherical harmonic modes, keyed by (ell, m), in code units rh/M
# t is the time array, in units of M
t, h, dyn = sur(q, chiA, chiB, dt=dt, f_low=f_low)   # dyn is always None for this model
```

### Physical units

This uses MKS (SI) units, with mass given in solar masses ($M_{\odot}$) and distance in
megaparsecs ($\mathrm{Mpc}$). `M` and `dist_mpc` must be given together, with `units='mks'`. `h`
is returned as the physical strain at that mass and distance; `t` is returned in
seconds.

```python
q = 4                      # mass ratio, mA/mB >= 1 (dimensionless)
chiA = [0, 0, 0.5]         # dimensionless spin of the heavier BH (z-component only)
chiB = [0, 0, -0.7]        # dimensionless spin of the lighter BH (z-component only)
M = 60                     # total mass, in solar masses
dist_mpc = 100             # distance to the source, in megaparsecs
dt = 1./4096               # timestep size, in seconds
f_low = 20                 # initial frequency of the (2,2) mode, in Hz

# h is a dict of spin-weighted spherical harmonic modes, keyed by (ell, m), as physical strain
# t is the time array, in seconds
t, h, dyn = sur(q, chiA, chiB, M=M, dist_mpc=dist_mpc, dt=dt, f_low=f_low, units='mks')
```

## `sur(...)` parameter reference

`NRHybSur3dq8` is evaluated by calling the loaded surrogate object directly:
`t, h, dyn = sur(q, chiA0, chiB0, ...)`. The full set of arguments is documented
below.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | float | *required* | Mass ratio, $m_A/m_B \ge 1$. |
| `chiA0` | array_like | *required* | Spin vector of the heavier BH, $[0, 0, \chi_{Az}]$. See [Spin convention](#spin-convention). |
| `chiB0` | array_like | *required* | Spin vector of the lighter BH, $[0, 0, \chi_{Bz}]$. Same convention as `chiA0`. |
| `M` | float | `None` | Total mass, in solar masses. Give together with `dist_mpc`, or not at all. |
| `dist_mpc` | float | `None` | Distance to the binary, in Mpc. Give together with `M`, or not at all. |
| `f_low` | float | *required* | Initial frequency of the (2,2) mode. See [Frequency parameters](#frequency-parameters). |
| `f_ref` | float | `f_low` | Reference frequency at which the reference frame is defined. See [Frequency parameters](#frequency-parameters). |
| `dt` | float | internal grid | Time step. If neither `dt` nor `times` is given, the surrogate's internal, nonuniformly sampled time grid is returned. Mutually exclusive with `times`. |
| `df` | float | — | **Not supported**: `NRHybSur3dq8` is a time-domain model; passing this raises an error. |
| `times` | array_like | `None` | Explicit time samples. Must lie within the waveform's time span. Mutually exclusive with `dt`. |
| `freqs` | array_like | — | **Not supported**: `NRHybSur3dq8` is a time-domain model; passing this raises an error. |
| `mode_list` | list of tuple | `None` (all modes) | List of $(\ell, m)$ modes with $m \ge 0$ to include, e.g. `[(2, 2), (3, 3)]`; the matching $m<0$ modes are added automatically. Mutually exclusive with `ellMax`. |
| `ellMax` | int | $5$ (all modes) | Maximum $\ell$ to include; all available $m$ for each $\ell$ are included automatically. Mutually exclusive with `mode_list`. |
| `inclination` | float | `None` | Inclination angle. If unset, `h` is returned as a dict of modes instead of a combined strain. |
| `phi_ref` | float | `0` | Azimuthal angle on the sky (LAL convention). |
| `precessing_opts` | dict | — | **Not supported**: `NRHybSur3dq8` is a nonprecessing model; passing anything but `None` raises an error. |
| `tidal_opts` | dict | — | **Not supported**: `NRHybSur3dq8` is not a tidal model; passing anything but `None` raises an error. See [NRHybSur3dq8Tidal](NRHybSur3dq8Tidal.md). |
| `par_dict` | dict | — | **Not supported**: not used by `NRHybSur3dq8`; passing anything but `None` raises an error. |
| `units` | `'dimensionless'` \| `'mks'` | `'dimensionless'` | Unit system for all quantities above. See [Units](#units). |
| `skip_param_checks` | bool | `False` | Skip all input validation checks. This is primarily intended to permit extrapolation outside the [validity region](#parameter-space-of-validity). |
| `taper_end_duration` | float | `None` | Taper the last `taper_end_duration` (in $M$) of the waveform. |

### Spin convention

`chiA0`/`chiB0` are dimensionless spin vectors $[\chi_x, \chi_y, \chi_z]$, where
$\chi_z = \vec\chi \cdot \hat L$ is the component along the orbital angular
momentum. Since this is an aligned-spin model, only $\chi_z$ is used: the in-plane
components $\chi_x, \chi_y$ must be zero, and nonzero values raise an error.

!!! warning
    With `skip_param_checks=True`, nonzero in-plane spin components are
    **silently ignored** rather than rejected.

### Frequency parameters

- `f_low` is the initial frequency of the (2,2) mode; the waveform is truncated
  so that it begins (approximately) at this frequency. It must lie between the
  surrogate's starting frequency and the peak frequency, otherwise an error is
  raised. Because the model is hybridized, its starting frequency is very low:
  roughly $7\times10^{-5}$–$9\times10^{-5}$ cycles/$M$ for the (2,2) mode,
  depending on $q$ and the spins (about $14$–$18\,\mathrm{Hz}$ for $M = 1\,M_\odot$).
- `f_low=0` returns the entire surrogate waveform, which starts about
  $5\times10^{8}\,M$ before the peak. Combined with a small `dt` (like $0.1M$),
  this leads to very expensive evaluation and very large output arrays.
- `f_ref` sets the reference epoch at which the reference frame is defined:
  the waveform is aligned so that the orbital phase is zero (the heavier BH is on
  the $+x$ axis) when the (2,2) mode frequency equals `f_ref`. Defaults to
  `f_low`. Must satisfy `f_ref >= f_low` and lie below the peak frequency. Since
  the spins are aligned and do not evolve, `f_ref` only sets the orientation of
  the reference frame, not the spins.
- Units: cycles/$M$ if `units='dimensionless'`, Hz if `units='mks'`.

### Units

- `'dimensionless'`: `dt`/`times` in $M$; `f_low`/`f_ref` in
  cycles/$M$; `M` and `dist_mpc` must be `None`. Waveform and domain are
  returned dimensionless.
- `'mks'`: `dt`/`times` in seconds; `f_low`/`f_ref` in Hz; `M` and
  `dist_mpc` must be given. Waveform and domain are returned in MKS units.

### Returns

`sur(...)` returns `domain, h, dynamics`:

| Name | Type | Description |
|---|---|---|
| `domain` | ndarray | Time samples for `h`. $t=0$ is set at the waveform peak. |
| `h` | ndarray or dict | If `inclination` is set: the complex strain $h = h_+ - i h_\times$ at $(\iota, \pi/2 - \phi_\mathrm{ref})$, including the $m<0$ modes. Otherwise: a dict of modes keyed by `(ell, m)`, e.g. `h[(2, 2)]`, including the $m<0$ modes. Physical units if `M`/`dist_mpc` given, else code units $rh/M$. |
| `dynamics` | `None` | Always `None`, since this is a nonprecessing model. |

!!! note "Reference frame convention"
    The reference (inertial) frame is defined at the reference epoch (set by
    `f_ref`): $+z$ is along the orbital angular momentum; $+x$ is along the
    separation vector from the lighter to the heavier BH; $y$ completes the
    right-handed triad. If `inclination`/`phi_ref` are given, the waveform is
    evaluated at $(\iota, \pi/2-\phi_\mathrm{ref})$ in this frame (the LAL
    convention; see LIGO DCC document T1800226 for the frame diagram).

## Full API reference

See the [`NRHybSur3dq8` API reference](../api/surrogate.md#gwsurrogate.surrogate.NRHybSur3dq8)
for the complete class listing.

## See also

- [`tutorial/website/NRHybSur3dq8.ipynb`](../tutorial/website/NRHybSur3dq8.ipynb):
  full worked tutorial.
- [NRHybSur3dq8_CCE](NRHybSur3dq8_CCE.md): the same parameter space, trained on
  CCE waveforms that include memory effects.
- [NRHybSur3dq8Tidal](NRHybSur3dq8Tidal.md): tidal extension for binary neutron
  stars and black hole–neutron star binaries.
- [NRHybSur2dq15](NRHybSur2dq15.md): aligned-spin model reaching higher mass ratios.
