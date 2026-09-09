# NRSur7dq4v2

[![arXiv:2609.07873](https://img.shields.io/badge/arXiv-2609.07873-B31B1B.svg)](https://arxiv.org/abs/2609.07873)
[![Data](https://img.shields.io/badge/data-Zenodo-1682D4.svg)](https://zenodo.org/records/22257361)

A domain-decomposed extension of [NRSur7dq4](NRSur7dq4.md) with
improved merger-ringdown accuracy, for generically precessing binary black holes.
See [arXiv:2609.07873](https://arxiv.org/abs/2609.07873) for details of the basis construction, truncation method, and
accuracy studies.

## Parameter space of validity

| Quantity | Trained / tested range | Soft limit | Hard limit |
|---|---|---|---|
| Mass ratio $q = m_A/m_B \ge 1$ | $q \in [1, 4]$ | $q \approx 4.01$ | $q \approx 6.01$ |
| Dimensionless spin magnitudes $\lvert \chi_A \rvert, \lvert \chi_B \rvert$ | $[0.0, 0.8]$ (generic direction) | $\approx 0.801$ | $1$ |
| Harmonic modes $h_{\ell m}$ | $\begin{aligned} 2 \le\ &\ell\ \le 5 \\ -\ell \le\ &m \le \ell \end{aligned}$ | — | — |

The model is trained within the range shown above, but can be evaluated up to
the hard limits; extrapolation behavior in that outer region is essentially
identical to `NRSur7dq4`; see [arXiv:2609.07873](https://arxiv.org/abs/2609.07873) for extrapolation tests and
limitations.

Crossing a **soft limit** prints a warning but still evaluates the surrogate;
crossing a **hard limit** raises an error and blocks evaluation.

## Loading the surrogate

The surrogate data only needs to be downloaded and loaded once per script/session.

```python
import gwsurrogate

gwsurrogate.catalog.pull('NRSur7dq4v2')   # one-time download
sur = gwsurrogate.LoadSurrogate('NRSur7dq4v2')
```

`LoadSurrogate` also accepts an optional `model_preset` argument, an advanced,
opt-in tradeoff of accuracy for speed; see [Advanced usage](#advanced-usage).

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
chiA = [-0.2, 0.4, 0.1]     # dimensionless spin of the heavier BH
chiB = [-0.5, 0.2, -0.4]    # dimensionless spin of the lighter BH
dt = 0.1                    # timestep size, in units of M
f_low = 0                   # initial frequency, in cycles/M; 0 returns the full surrogate

# h is a dict of spin-weighted spherical harmonic modes, keyed by (ell, m), in code units rh/M
# t is the time array, in units of M
t, h, dyn = sur(q, chiA, chiB, dt=dt, f_low=f_low)
```

### Physical units

This uses MKS (SI) units, with mass given in solar masses ($M_{\odot}$) and distance in
megaparsecs ($\mathrm{Mpc}$). `M` and `dist_mpc` must be given together, with `units='mks'`. `h`
is returned as the physical strain at that mass and distance; `t` is returned in
seconds.

```python
q = 4                      # mass ratio, mA/mB >= 1 (dimensionless)
chiA = [-0.2, 0.4, 0.1]    # dimensionless spin of the heavier BH
chiB = [-0.5, 0.2, -0.4]   # dimensionless spin of the lighter BH
M = 70                     # total mass, in solar masses
dist_mpc = 100             # distance to the source, in megaparsecs
dt = 1./4096               # timestep size, in seconds
f_low = 0                  # initial frequency, in Hz; 0 returns the full surrogate

# h is a dict of spin-weighted spherical harmonic modes, keyed by (ell, m), as physical strain
# t is the time array, in seconds
t, h, dyn = sur(q, chiA, chiB, M=M, dist_mpc=dist_mpc, dt=dt, f_low=f_low, units='mks')
```

## `sur(...)` parameter reference

`NRSur7dq4v2` is evaluated by calling the loaded surrogate object directly:
`t, h, dyn = sur(q, chiA0, chiB0, ...)`. The full set of arguments is documented
below.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | float | *required* | Mass ratio, $m_A/m_B \ge 1$. |
| `chiA0` | array_like | *required* | Spin vector of the heavier BH at the reference epoch. See [Spin convention](#spin-convention). |
| `chiB0` | array_like | *required* | Spin vector of the lighter BH at the reference epoch. Same convention as `chiA0`. |
| `M` | float | `None` | Total mass, in solar masses. Give together with `dist_mpc`, or not at all. |
| `dist_mpc` | float | `None` | Distance to the binary, in Mpc. Give together with `M`, or not at all. |
| `f_low` | float | *required* | Initial frequency of the (2,2) mode; `0` returns the full waveform. See [Frequency parameters](#frequency-parameters). |
| `f_ref` | float | `f_low` | Reference frequency setting the reference epoch/frame. See [Frequency parameters](#frequency-parameters). |
| `dt` | float | internal grid | Time step. Mutually exclusive with `times`. |
| `df` | float | — | **Not supported**: `NRSur7dq4v2` is a time-domain model; passing this raises an error. |
| `times` | array_like | `None` | Explicit time samples. Mutually exclusive with `dt`. |
| `freqs` | array_like | — | **Not supported**: `NRSur7dq4v2` is a time-domain model; passing this raises an error. |
| `mode_list` | list of tuple | — | **Not supported**: `NRSur7dq4v2` is a precessing model; passing this raises an error. Use `ellMax` instead. |
| `ellMax` | int | $5$ (all modes) | Maximum $\ell$ to include; all $m$ for that $\ell$ are included automatically. |
| `inclination` | float | `None` | Inclination angle. If unset, `h` is returned as a dict of modes instead of a combined strain. |
| `phi_ref` | float | `0` | Azimuthal angle on the sky (LAL convention). |
| `precessing_opts` | dict | `None` | Precessing-model options. See [Precessing options](#precessing-options). |
| `tidal_opts` | dict | — | **Not supported**: `NRSur7dq4v2` is not a tidal model; passing this raises an error. |
| `par_dict` | dict | — | **Not supported**: not used by `NRSur7dq4v2`; passing anything but `None` raises an error. |
| `units` | `'dimensionless'` \| `'mks'` | `'dimensionless'` | Unit system for all quantities above. See [Units](#units). |
| `skip_param_checks` | bool | `False` | Skip range checks, forcing evaluation outside the [validity region](#parameter-space-of-validity). |
| `taper_end_duration` | float | `None` | Taper the last `taper_end_duration` (in $M$) of the waveform. |

### Spin convention

`chiA0`/`chiB0` are dimensionless spin vectors $[\chi_x, \chi_y, \chi_z]$ at the
reference epoch, in the LAL convention (frame-independent, via vector inner
products):

- $\chi_z = \vec\chi \cdot \hat L$, with $\hat L$ the orbital angular momentum
  direction at the epoch.
- $\chi_x = \vec\chi \cdot \hat n$, with $\hat n$ the separation vector from the
  lighter to the heavier body at the epoch.
- $\chi_y = \vec\chi \cdot (\hat L \times \hat n)$.

This is equivalent to specifying spins in the coorbital frame used in the
surrogate papers.

### Frequency parameters

- `f_low` is (approximately) twice the initial orbital frequency in the
  coprecessing frame. For `NRSur7dq4v2`, which is already a short waveform, `f_low=0`
  is recommended, returning the entire surrogate.
- `f_ref` sets the reference epoch at which the reference frame is defined and
  the spins are specified. Defaults to `f_low` if not given. Capped at a
  reference orbital frequency of $M\Omega_\mathrm{orb}^\mathrm{ref} \le 0.201$.
- Units: cycles/$M$ if `units='dimensionless'`, Hz if `units='mks'`.

### Precessing options

`precessing_opts` is a dict with any of the following keys:

| Key | Default | Description |
|---|---|---|
| `init_orbphase` | `0` | Orbital phase in the coprecessing frame at the reference epoch. |
| `init_quat` | `None` | Length-4 unit quaternion rotating the coprecessing frame into the inertial frame at the reference epoch. |
| `return_dynamics` | `False` | Also return the frame/spin dynamics (see [Returns](#returns) below). |
| `debug_work_stats` | `False` | Print the number of evaluated coorbital data pieces and scalar-fit nodes (performance diagnostics). |

Example: `precessing_opts = {'init_orbphase': 0, 'init_quat': [1, 0, 0, 0], 'return_dynamics': True}`

### Units

- `'dimensionless'`: `dt`/`times` in $M$; `f_low`/`f_ref`/`df`/`freqs` in
  cycles/$M$; `M` and `dist_mpc` must be `None`. Waveform and domain are
  returned dimensionless.
- `'mks'`: `dt`/`times` in seconds; `f_low`/`f_ref`/`df`/`freqs` in Hz; `M` and
  `dist_mpc` must be given. Waveform and domain are returned in MKS units.

### Returns

`sur(...)` returns `domain, h, dynamics`:

| Name | Type | Description |
|---|---|---|
| `domain` | ndarray | Time (or frequency) samples for `h`/`dynamics`. $t=0$ is set at the waveform peak. |
| `h` | ndarray or dict | If `inclination` is set: the complex strain $h = h_+ - i h_\times$ at $(\iota, \pi/2 - \phi_\mathrm{ref})$. Otherwise: a dict of modes keyed by `(ell, m)`, e.g. `h[(2, 2)]`. Physical units if `M`/`dist_mpc` given, else code units $rh/M$. |
| `dynamics` | dict or `None` | Frame/spin dynamics, only if `precessing_opts['return_dynamics']` is `True`. See below. |

The `dynamics` dict (with $L$ = `len(domain)`) contains:

| Key | Shape | Description |
|---|---|---|
| `q_copr` | `(4, L)` | Quaternion of the coprecessing frame. |
| `orbphase` | `(L,)` | Orbital phase in the coprecessing frame. |
| `chiA` | `(L, 3)` | Inertial-frame spin of the heavier BH. |
| `chiB` | `(L, 3)` | Inertial-frame spin of the lighter BH. |

!!! note "Reference frame convention"
    The reference (inertial) frame is defined at the reference epoch (set by
    `f_ref`): $+z$ is along the orbital angular momentum; $+x$ is along the
    separation vector from the lighter to the heavier BH; $y$ completes the
    right-handed triad. If `inclination`/`phi_ref` are given, the waveform is
    evaluated at $(\iota, \pi/2-\phi_\mathrm{ref})$ in this frame (the LAL
    convention; see LIGO DCC document T1800226 for the frame diagram).

## Advanced usage

/// details | Basis-size presets (optional, advanced)
    type: warning

This is an opt-in, advanced feature. Most users should stick with the default,
full-accuracy model; only reach for a preset if you specifically need faster
evaluation and have validated the accuracy tradeoff for your own use case.

`NRSur7dq4v2` supports basis-size presets that retain fewer basis elements for
selected coorbital data pieces, trading some accuracy for faster evaluation. The
packaged `"Fast"` preset gives the tested accuracy/cost tradeoff from
[arXiv:2609.07873](https://arxiv.org/abs/2609.07873):

```python
sur_fast = gwsurrogate.LoadSurrogate('NRSur7dq4v2', model_preset="Fast")
```

Preset definitions live in `gwsurrogate.new._model_presets.MODEL_PRESETS`; see
[`tutorial/website/NRSur7dq4v2.ipynb`](../tutorial/website/NRSur7dq4v2.ipynb) for
how to register a custom preset.
///

## Full API reference

See the [`NRSur7dq4v2` API reference](../api/surrogate.md#gwsurrogate.surrogate.NRSur7dq4v2)
for the complete class listing, including `coorbital_basis_sizes`.

## See also

- [`tutorial/website/NRSur7dq4v2.ipynb`](../tutorial/website/NRSur7dq4v2.ipynb):
  full worked tutorial, including comparison against `NRSur7dq4` and custom preset
  construction.
- [NRSur7dq4](NRSur7dq4.md): the model this one extends.
