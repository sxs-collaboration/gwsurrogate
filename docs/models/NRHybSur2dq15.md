# NRHybSur2dq15

[![arXiv:2203.10109](https://img.shields.io/badge/arXiv-2203.10109-B31B1B.svg)](https://arxiv.org/abs/2203.10109)
[![Data](https://img.shields.io/badge/data-Zenodo-1682D4.svg)](https://zenodo.org/record/6726994)

A surrogate model for **aligned-spin** (nonprecessing) binary black holes with
**high mass ratios** and a spin on the heavier black hole only. It was trained on
numerical relativity (NR) waveforms that were hybridized with effective-one-body
(EOB) waveforms at early times, so the model spans the entire LIGO frequency band
for stellar-mass binaries. See [arXiv:2203.10109](https://arxiv.org/abs/2203.10109) for details of the
model construction and accuracy studies.

## Parameter space of validity

| Quantity | Trained / tested range | Soft limit | Hard limit |
|---|---|---|---|
| Mass ratio $q = m_A/m_B \ge 1$ | $q \in [1, 15]$ | $q \approx 15.01$ | $q \approx 20.1$ |
| Aligned spin of the heavier BH $\chi_{Az}$ | $[-0.5, 0.5]$ | $\lvert\chi_A\rvert = 0.5$ | $\lvert\chi_A\rvert \approx 0.71$ |
| Spin of the lighter BH $\chi_B$ | $0$ | — | must be exactly $0$ |
| Harmonic modes $h_{\ell m}$ | $(2,2), (2,1), (3,3), (4,4), (5,5)$, and the corresponding $m<0$ modes | — | — |

The model is trained within the range shown above, but produces reasonable
waveforms up to the hard limits; see [arXiv:2203.10109](https://arxiv.org/abs/2203.10109).

Crossing a **soft limit** prints a warning but still evaluates the surrogate;
crossing a **hard limit** raises an error and blocks evaluation.

Only the $m \ge 0$ modes are modeled; the $m<0$ modes are deduced from the
orbital-plane symmetry of nonprecessing binaries,
$h_{\ell,-m} = (-1)^\ell\, h_{\ell m}^*$.

## Loading the surrogate

The surrogate data only needs to be downloaded and loaded once per script/session.

```python
import gwsurrogate

gwsurrogate.catalog.pull('NRHybSur2dq15')   # one-time download
sur = gwsurrogate.LoadSurrogate('NRHybSur2dq15')
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
q = 12                      # mass ratio, mA/mB >= 1 (dimensionless)
chiA = [0, 0, 0.4]          # dimensionless spin of the heavier BH (z-component only)
chiB = [0, 0, 0]            # the lighter BH must be nonspinning
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
q = 12                     # mass ratio, mA/mB >= 1 (dimensionless)
chiA = [0, 0, 0.4]         # dimensionless spin of the heavier BH (z-component only)
chiB = [0, 0, 0]           # the lighter BH must be nonspinning
M = 60                     # total mass, in solar masses
dist_mpc = 100             # distance to the source, in megaparsecs
dt = 1./4096               # timestep size, in seconds
f_low = 20                 # initial frequency of the (2,2) mode, in Hz

# h is a dict of spin-weighted spherical harmonic modes, keyed by (ell, m), as physical strain
# t is the time array, in seconds
t, h, dyn = sur(q, chiA, chiB, M=M, dist_mpc=dist_mpc, dt=dt, f_low=f_low, units='mks')
```

## `sur(...)` parameter reference

`NRHybSur2dq15` is evaluated by calling the loaded surrogate object directly:
`t, h, dyn = sur(q, chiA0, chiB0, ...)`. The full set of arguments is documented
below.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | float | *required* | Mass ratio, $m_A/m_B \ge 1$. |
| `chiA0` | array_like | *required* | Spin vector of the heavier BH, $[0, 0, \chi_{Az}]$. See [Spin convention](#spin-convention). |
| `chiB0` | array_like | *required* | Spin vector of the lighter BH; must be `[0, 0, 0]`. |
| `M` | float | `None` | Total mass, in solar masses. Give together with `dist_mpc`, or not at all. |
| `dist_mpc` | float | `None` | Distance to the binary, in Mpc. Give together with `M`, or not at all. |
| `f_low` | float | *required* | Initial frequency of the (2,2) mode. See [Frequency parameters](#frequency-parameters). |
| `f_ref` | float | `f_low` | Reference frequency at which the reference frame is defined. See [Frequency parameters](#frequency-parameters). |
| `dt` | float | internal grid | Time step. If neither `dt` nor `times` is given, the surrogate's internal, nonuniformly sampled time grid is returned. Mutually exclusive with `times`. |
| `df` | float | — | **Not supported**: `NRHybSur2dq15` is a time-domain model; passing this raises an error. |
| `times` | array_like | `None` | Explicit time samples. Must lie within the waveform's time span. Mutually exclusive with `dt`. |
| `freqs` | array_like | — | **Not supported**: `NRHybSur2dq15` is a time-domain model; passing this raises an error. |
| `mode_list` | list of tuple | `None` (all modes) | List of $(\ell, m)$ modes with $m \ge 0$ to include, e.g. `[(2, 2), (3, 3)]`; the matching $m<0$ modes are added automatically. Mutually exclusive with `ellMax`. |
| `ellMax` | int | $5$ (all modes) | Maximum $\ell$ to include; all available $m$ for each $\ell$ are included automatically. Mutually exclusive with `mode_list`. |
| `inclination` | float | `None` | Inclination angle. If unset, `h` is returned as a dict of modes instead of a combined strain. |
| `phi_ref` | float | `0` | Azimuthal angle on the sky (LAL convention). |
| `precessing_opts` | dict | — | **Not supported**: `NRHybSur2dq15` is a nonprecessing model; passing anything but `None` raises an error. |
| `tidal_opts` | dict | — | **Not supported**: `NRHybSur2dq15` is not a tidal model; passing anything but `None` raises an error. |
| `par_dict` | dict | — | **Not supported**: not used by `NRHybSur2dq15`; passing anything but `None` raises an error. |
| `units` | `'dimensionless'` \| `'mks'` | `'dimensionless'` | Unit system for all quantities above. See [Units](#units). |
| `skip_param_checks` | bool | `False` | Skip all input validation checks. This is primarily intended to permit extrapolation outside the [validity region](#parameter-space-of-validity). |
| `taper_end_duration` | float | `None` | Taper the last `taper_end_duration` (in $M$) of the waveform. |

### Spin convention

`chiA0`/`chiB0` are dimensionless spin vectors $[\chi_x, \chi_y, \chi_z]$, where
$\chi_z = \vec\chi \cdot \hat L$ is the component along the orbital angular
momentum.

- Only $\chi_{Az}$ is used. The in-plane components $\chi_{Ax}, \chi_{Ay}$ must be
  zero, and nonzero values raise an error.
- The lighter BH must be nonspinning: a nonzero $\chi_{Bz}$ raises an error
  (even with `skip_param_checks=True`), as do nonzero in-plane components.

!!! warning
    With `skip_param_checks=True`, nonzero in-plane spin components are
    **silently ignored** rather than rejected.

### Frequency parameters

- `f_low` is the initial frequency of the (2,2) mode; the waveform is truncated
  so that it begins (approximately) at this frequency. It must lie between the
  surrogate's starting frequency and the peak frequency, otherwise an error is
  raised. Because the model is hybridized, its starting frequency is low:
  roughly $2\times10^{-4}$–$4\times10^{-4}$ cycles/$M$ for the (2,2) mode,
  increasing with $q$ (about $40$–$75\,\mathrm{Hz}$ for $M = 1\,M_\odot$, or
  $4$–$7\,\mathrm{Hz}$ for $M = 10\,M_\odot$).
- `f_low=0` returns the entire surrogate waveform, which starts about
  $2\times10^{7}\,M$ before the peak. Combined with a small `dt` (like $0.1M$),
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

See the [`NRHybSur2dq15` API reference](../api/surrogate.md#gwsurrogate.surrogate.NRHybSur2dq15)
for the complete class listing.

## See also

- [`tutorial/website/NRHybSur2dq15.ipynb`](../tutorial/website/NRHybSur2dq15.ipynb):
  full worked tutorial.
- [NRHybSur3dq8](NRHybSur3dq8.md): aligned-spin model with spins on both black
  holes, for $q \le 8$.
- [BHPTNRSur1dq1e4](BHPTNRSur1dq1e4.md): nonspinning model reaching extreme mass
  ratios.
