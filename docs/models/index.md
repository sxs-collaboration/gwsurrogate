# Models

Here you can find a list of available surrogate models. Each model page describes
the model's parameter space of validity, how to load and evaluate it, and the
full set of evaluation arguments.

| Model | Binary type | Mass ratio $q$ (trained) | Spins (trained) | Modes | Training data |
|---|---|---|---|---|---|
| [NRSur7dq4v2](NRSur7dq4v2.md) | Precessing BBH | $[1, 4]$ | $\lvert\chi\rvert \le 0.8$, generic | $\ell \le 5$ | NR |
| [NRSur7dq4](NRSur7dq4.md) | Precessing BBH | $[1, 4]$ | $\lvert\chi\rvert \le 0.8$, generic | $\ell \le 4$ | NR |
| [SEOBNRv4PHMSur](SEOBNRv4PHMSur.md) | | | | | |
| [NRHybSur3dq8](NRHybSur3dq8.md) | Aligned-spin BBH | $[1, 8]$ | $\lvert\chi_z\rvert \le 0.8$ | $\ell \le 5$ (11 modes) | NR hybridized with PN/EOB |
| [NRHybSur3dq8_CCE](NRHybSur3dq8_CCE.md) | Aligned-spin BBH, with memory | $[1, 8]$ | $\lvert\chi_z\rvert \le 0.8$ | $\ell \le 5$ (10 modes) | CCE NR hybridized with PN/EOB |
| [NRHybSur2dq15](NRHybSur2dq15.md) | Aligned-spin BBH | $[1, 15]$ | $\lvert\chi_{Az}\rvert \le 0.5$, $\chi_B = 0$ | $\ell \le 5$ (5 modes) | NR hybridized with EOB |
| [NRHybSur3dq8Tidal](NRHybSur3dq8Tidal.md) | | | | | |
| [BHPTNRSur1dq1e4](BHPTNRSur1dq1e4.md) | | | | | |
| [EMRISur1dq1e4](EMRISur1dq1e4.md) | | | | | |

The number of modes counts the $m \ge 0$ modes; the corresponding $m<0$ modes are
also returned. Most models can be evaluated somewhat outside their training
range; see each model's page for its soft (warning) and hard (error) limits.

All models except `BHPTNRSur1dq1e4` and `EMRISur1dq1e4` are loaded with
`gwsurrogate.LoadSurrogate` and share a common calling interface;
`BHPTNRSur1dq1e4` and `EMRISur1dq1e4` are loaded with
`gwsurrogate.EvaluateSurrogate`, which has a different interface.
