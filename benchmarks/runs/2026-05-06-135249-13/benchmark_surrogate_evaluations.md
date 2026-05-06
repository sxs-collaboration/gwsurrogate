# GWSurrogate Evaluation Timing

Generated: 2026-05-06T13:52:43.718306+00:00

Times below are seconds per model evaluation. Raw repeats and context are in the JSON output.

PNG timing table: `test/benchmark_surrogate_evaluations.png`

## Summary

### master

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.069471` s, median `0.0697252` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0558896` s, median `0.0563281` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.10119` s, median `0.101296` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0777993` s, median `0.0782044` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.251036` s, median `0.252819` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0821186` s, median `0.0823916` s
- `dt=0.5 M`, `f_low=0`: best `0.0733211` s, median `0.0735748` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0435992` s, median `0.0442498` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.106833` s, median `0.108873` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0824946` s, median `0.0838826` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.163167` s, median `0.164001` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.113743` s, median `0.114643` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.434455` s, median `0.438791` s
- `dt=0.1 M`, `f_low=0.01`: best `0.12595` s, median `0.126284` s
- `dt=0.5 M`, `f_low=0`: best `0.124743` s, median `0.125268` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0683003` s, median `0.0685989` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.475449` s, median `0.480982` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.150157` s, median `0.151253` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.82867` s, median `0.830369` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.171326` s, median `0.171985` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.18352` s, median `0.184449` s
- `dt=0.1 M`, `f_low=0.002`: best `4.07481` s, median `4.08405` s
- `dt=0.5 M`, `f_low=0.01`: best `0.1295` s, median `0.13177` s
- `dt=0.5 M`, `f_low=0.002`: best `0.933454` s, median `0.94461` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.188573` s, median `0.190743` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0392353` s, median `0.0396469` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.344089` s, median `0.347442` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.048668` s, median `0.0492048` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0532068` s, median `0.0544216` s
- `dt=0.1 M`, `f_low=0.002`: best `1.82073` s, median `1.83101` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0339759` s, median `0.0341763` s
- `dt=0.5 M`, `f_low=0.002`: best `0.390277` s, median `0.395068` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.442311` s, median `0.45086` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.142271` s, median `0.145626` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.748561` s, median `0.751565` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.171455` s, median `0.172009` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.184203` s, median `0.185295` s
- `dt=0.1 M`, `f_low=0.002`: best `3.70269` s, median `3.71034` s
- `dt=0.5 M`, `f_low=0.01`: best `0.128776` s, median `0.131154` s
- `dt=0.5 M`, `f_low=0.002`: best `0.855553` s, median `0.858951` s

### PR-71

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0224297` s, median `0.0237921` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0223928` s, median `0.0226448` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0254963` s, median `0.0258996` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0249739` s, median `0.0253301` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0202695` s, median `0.0203752` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0208726` s, median `0.021728` s
- `dt=0.5 M`, `f_low=0`: best `0.0168319` s, median `0.0169564` s
- `dt=0.5 M`, `f_low=0.01`: best `0.020205` s, median `0.0203306` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0332182` s, median `0.0336233` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0326882` s, median `0.0332327` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0345957` s, median `0.035061` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0340453` s, median `0.0341158` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0356202` s, median `0.036257` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0338845` s, median `0.0342351` s
- `dt=0.5 M`, `f_low=0`: best `0.0285915` s, median `0.0289919` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0326499` s, median `0.0328097` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0974453` s, median `0.0979153` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0634776` s, median `0.0642683` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.140828` s, median `0.142401` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0628348` s, median `0.06415` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0632135` s, median `0.0633746` s
- `dt=0.1 M`, `f_low=0.002`: best `0.348432` s, median `0.352146` s
- `dt=0.5 M`, `f_low=0.01`: best `0.062836` s, median `0.0650173` s
- `dt=0.5 M`, `f_low=0.002`: best `0.124013` s, median `0.124685` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0324726` s, median `0.0328322` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0190468` s, median `0.0192189` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0477002` s, median `0.0480926` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0195061` s, median `0.0198043` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0194968` s, median `0.0201414` s
- `dt=0.1 M`, `f_low=0.002`: best `0.178233` s, median `0.178855` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0180776` s, median `0.0189024` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0483478` s, median `0.0489413` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0999423` s, median `0.101304` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0745489` s, median `0.0751607` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.126864` s, median `0.128856` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0772598` s, median `0.0816623` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0787761` s, median `0.079066` s
- `dt=0.1 M`, `f_low=0.002`: best `0.334775` s, median `0.33902` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0724179` s, median `0.0742971` s
- `dt=0.5 M`, `f_low=0.002`: best `0.129503` s, median `0.130538` s

## Context

### master

- Git branch: `master`
- Git commit: `a84a5da1aa62624dd73c52103ab7fab6410bb32a`
- Git describe: `v1.1.8-11-ga84a5da`
- Python: `3.14.4 (main, Apr  8 2026, 02:27:22) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1010-azure x86_64`
- CPU count: `4`
- Conda env: `unknown`

### PR-71

- Git branch: `unknown`
- Git commit: `81b8ad1fead7691859429e2b67de3e1ce1ac55a9`
- Git describe: `fatal: not a git repository: gwsurrogate/eval_pysur/../../.git/modules/gwsurrogate/eval_pysur`
- Python: `3.14.4 (main, Apr  8 2026, 02:27:22) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1010-azure x86_64`
- CPU count: `4`
- Conda env: `unknown`

Git status:

```text
fatal: not a git repository: gwsurrogate/eval_pysur/../../.git/modules/gwsurrogate/eval_pysur
```

## Appendix

### Hardware Data

#### master

lscpu:

```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           48 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  4
On-line CPU(s) list:                     0-3
Vendor ID:                               AuthenticAMD
Model name:                              AMD EPYC 7763 64-Core Processor
CPU family:                              25
Model:                                   1
Thread(s) per core:                      2
Core(s) per socket:                      2
Socket(s):                               1
Stepping:                                1
BogoMIPS:                                4890.85
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves user_shstk clzero xsaveerptr rdpru arat npt nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload umip vaes vpclmulqdq rdpid fsrm
Virtualization:                          AMD-V
Hypervisor vendor:                       Microsoft
Virtualization type:                     full
L1d cache:                               64 KiB (2 instances)
L1i cache:                               64 KiB (2 instances)
L2 cache:                                1 MiB (2 instances)
L3 cache:                                32 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-3
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Vulnerable: Safe RET, no microcode
Vulnerability Spec store bypass:         Vulnerable
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Retpolines; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Vulnerable: No microcode
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Not affected
```

#### PR-71

lscpu:

```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           48 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  4
On-line CPU(s) list:                     0-3
Vendor ID:                               AuthenticAMD
Model name:                              AMD EPYC 7763 64-Core Processor
CPU family:                              25
Model:                                   1
Thread(s) per core:                      2
Core(s) per socket:                      2
Socket(s):                               1
Stepping:                                1
BogoMIPS:                                4890.85
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves user_shstk clzero xsaveerptr rdpru arat npt nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload umip vaes vpclmulqdq rdpid fsrm
Virtualization:                          AMD-V
Hypervisor vendor:                       Microsoft
Virtualization type:                     full
L1d cache:                               64 KiB (2 instances)
L1i cache:                               64 KiB (2 instances)
L2 cache:                                1 MiB (2 instances)
L3 cache:                                32 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-3
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Vulnerable: Safe RET, no microcode
Vulnerability Spec store bypass:         Vulnerable
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Retpolines; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Vulnerable: No microcode
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Not affected
```

### cProfile

#### master

##### NRSur7dq4 / mks_dt_0.000244140625_flow_0

```text
         28760 function calls in 0.079 seconds

   Ordered by: cumulative time
   List reduced from 95 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.079    0.079 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.079    0.079 surrogate.py:1721(__call__)
        1    0.001    0.001    0.079    0.079 precessing_surrogate.py:933(__call__)
      196    0.000    0.000    0.050    0.000 surrogate.py:85(_splinterp_Cwrapper)
      196    0.008    0.000    0.049    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.046    0.009 precessing_surrogate.py:849(splinterp_many)
      560    0.019    0.000    0.019    0.000 {built-in method builtins.min}
      561    0.019    0.000    0.019    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:387(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:105(rotateWaveform)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.006    0.006    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:621(_integrate_forward)
        3    0.000    0.000    0.007    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     1638    0.003    0.000    0.003    0.000 {built-in method numpy.array}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
      784    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:655(_integrate_backward)
      784    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
      588    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      784    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:340(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      201    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:334(get_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         28761 function calls in 0.065 seconds

   Ordered by: cumulative time
   List reduced from 95 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.065    0.065 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.064    0.064 surrogate.py:1721(__call__)
        1    0.001    0.001    0.064    0.064 precessing_surrogate.py:933(__call__)
      196    0.000    0.000    0.039    0.000 surrogate.py:85(_splinterp_Cwrapper)
      196    0.007    0.000    0.039    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.035    0.007 precessing_surrogate.py:849(splinterp_many)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:387(__call__)
      560    0.015    0.000    0.015    0.000 {built-in method builtins.min}
      561    0.014    0.000    0.014    0.000 {built-in method builtins.max}
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:105(rotateWaveform)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        1    0.005    0.005    0.007    0.007 precessing_surrogate.py:42(_wignerD_matrices)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:621(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      784    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:655(_integrate_backward)
     1638    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      784    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      588    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      784    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:340(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:334(get_omega)
      201    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         28760 function calls in 0.114 seconds

   Ordered by: cumulative time
   List reduced from 95 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.114    0.114 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.114    0.114 surrogate.py:1721(__call__)
        1    0.002    0.002    0.113    0.113 precessing_surrogate.py:933(__call__)
      196    0.000    0.000    0.084    0.000 surrogate.py:85(_splinterp_Cwrapper)
      196    0.011    0.000    0.083    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.079    0.016 precessing_surrogate.py:849(splinterp_many)
      560    0.034    0.000    0.034    0.000 {built-in method builtins.min}
      561    0.034    0.000    0.034    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:387(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:105(rotateWaveform)
        1    0.006    0.006    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.007    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:621(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:764(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     1638    0.003    0.000    0.003    0.000 {built-in method numpy.array}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      784    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:655(_integrate_backward)
      784    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      588    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      784    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
      201    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:340(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:334(get_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         28761 function calls in 0.087 seconds

   Ordered by: cumulative time
   List reduced from 95 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.087    0.087 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.087    0.087 surrogate.py:1721(__call__)
        1    0.001    0.001    0.086    0.086 precessing_surrogate.py:933(__call__)
      196    0.000    0.000    0.059    0.000 surrogate.py:85(_splinterp_Cwrapper)
      196    0.009    0.000    0.058    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.054    0.011 precessing_surrogate.py:849(splinterp_many)
      560    0.023    0.000    0.023    0.000 {built-in method builtins.min}
      561    0.023    0.000    0.023    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:387(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:105(rotateWaveform)
        1    0.006    0.006    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:621(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:764(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     1638    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      784    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:655(_integrate_backward)
      784    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
      588    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      784    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:340(_get_t_from_omega)
      201    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:334(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         21557 function calls in 0.262 seconds

   Ordered by: cumulative time
   List reduced from 83 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.262    0.262 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.262    0.262 surrogate.py:1721(__call__)
        1    0.004    0.004    0.262    0.262 precessing_surrogate.py:933(__call__)
        5    0.000    0.000    0.233    0.047 precessing_surrogate.py:849(splinterp_many)
       53    0.000    0.000    0.230    0.004 surrogate.py:85(_splinterp_Cwrapper)
       53    0.023    0.000    0.230    0.004 spline_interp_Cwrapper.py:39(interpolate)
      261    0.102    0.000    0.102    0.000 {built-in method builtins.min}
      262    0.102    0.000    0.102    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:105(rotateWaveform)
        1    0.006    0.006    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:387(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:621(_integrate_forward)
      238    0.001    0.000    0.008    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:179(_eval_vector_fit)
     1257    0.005    0.000    0.005    0.000 {built-in method numpy.array}
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2390    0.001    0.000    0.004    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
     2390    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
       58    0.002    0.000    0.002    0.000 {built-in method numpy.zeros}
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      159    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
      212    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:556(_initial_RK4)
      212    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:703(_assemble_mode_pair)
      212    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:827(rotate_spin)
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:837(coorb_spins_from_copr_spins)
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         29367 function calls in 0.092 seconds

   Ordered by: cumulative time
   List reduced from 93 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.092    0.092 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.092    0.092 surrogate.py:1721(__call__)
        1    0.001    0.001    0.092    0.092 precessing_surrogate.py:933(__call__)
      196    0.000    0.000    0.066    0.000 surrogate.py:85(_splinterp_Cwrapper)
      196    0.009    0.000    0.066    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.062    0.012 precessing_surrogate.py:849(splinterp_many)
      560    0.027    0.000    0.027    0.000 {built-in method builtins.min}
      561    0.027    0.000    0.027    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:387(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:842(inertial_waveform_modes)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:105(rotateWaveform)
        3    0.000    0.000    0.007    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:621(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:655(_integrate_backward)
     2876    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      644    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      784    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
      784    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1708    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:340(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      588    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      784    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      201    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         21557 function calls in 0.082 seconds

   Ordered by: cumulative time
   List reduced from 83 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.082    0.082 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.082    0.082 surrogate.py:1721(__call__)
        1    0.001    0.001    0.082    0.082 precessing_surrogate.py:933(__call__)
        5    0.000    0.000    0.058    0.012 precessing_surrogate.py:849(splinterp_many)
       53    0.000    0.000    0.058    0.001 surrogate.py:85(_splinterp_Cwrapper)
       53    0.007    0.000    0.058    0.001 spline_interp_Cwrapper.py:39(interpolate)
      262    0.025    0.000    0.025    0.000 {built-in method builtins.max}
      261    0.025    0.000    0.025    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:387(__call__)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:621(_integrate_forward)
      238    0.001    0.000    0.008    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:105(rotateWaveform)
        1    0.005    0.005    0.007    0.007 precessing_surrogate.py:42(_wignerD_matrices)
      714    0.001    0.000    0.006    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
     2390    0.001    0.000    0.005    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2390    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     1257    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
      212    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:556(_initial_RK4)
      159    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      212    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
       58    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:703(_assemble_mode_pair)
      212    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:827(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:837(coorb_spins_from_copr_spins)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         29367 function calls in 0.068 seconds

   Ordered by: cumulative time
   List reduced from 93 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.068    0.068 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.068    0.068 surrogate.py:1721(__call__)
        1    0.000    0.000    0.068    0.068 precessing_surrogate.py:933(__call__)
      196    0.000    0.000    0.040    0.000 surrogate.py:85(_splinterp_Cwrapper)
      196    0.007    0.000    0.039    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.035    0.007 precessing_surrogate.py:849(splinterp_many)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:387(__call__)
      561    0.015    0.000    0.015    0.000 {built-in method builtins.max}
      560    0.015    0.000    0.015    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.011    0.011 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.011    0.011 precessing_surrogate.py:105(rotateWaveform)
        1    0.007    0.007    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:764(__call__)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:621(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:655(_integrate_backward)
     2876    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      644    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.002    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
      784    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
      784    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1708    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:340(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
      588    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      784    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      201    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         45328 function calls in 0.123 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.123    0.123 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.123    0.123 surrogate.py:1721(__call__)
        1    0.000    0.000    0.123    0.123 precessing_surrogate.py:1263(__call__)
        1    0.001    0.001    0.122    0.122 precessing_surrogate.py:933(__call__)
      218    0.000    0.000    0.081    0.000 surrogate.py:85(_splinterp_Cwrapper)
      218    0.012    0.000    0.080    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.076    0.015 precessing_surrogate.py:849(splinterp_many)
      725    0.032    0.000    0.032    0.000 {built-in method builtins.min}
      726    0.032    0.000    0.032    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:387(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:621(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:105(rotateWaveform)
     5135    0.003    0.000    0.013    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
     5135    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.007    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:655(_integrate_backward)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
     2610    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      872    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
      872    0.002    0.000    0.002    0.000 __init__.py:613(cast)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      654    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:340(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
      224    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      872    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         45329 function calls in 0.098 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.098    0.098 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.098    0.098 surrogate.py:1721(__call__)
        1    0.000    0.000    0.097    0.097 precessing_surrogate.py:1263(__call__)
        1    0.001    0.001    0.097    0.097 precessing_surrogate.py:933(__call__)
      218    0.000    0.000    0.056    0.000 surrogate.py:85(_splinterp_Cwrapper)
      218    0.009    0.000    0.056    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.052    0.010 precessing_surrogate.py:849(splinterp_many)
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:387(__call__)
      725    0.022    0.000    0.022    0.000 {built-in method builtins.min}
      726    0.021    0.000    0.021    0.000 {built-in method builtins.max}
      546    0.002    0.000    0.020    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:621(_integrate_forward)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:179(_eval_vector_fit)
     5135    0.003    0.000    0.013    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:105(rotateWaveform)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:655(_integrate_backward)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
     2610    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      872    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      872    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      654    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:340(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      224    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      872    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
      546    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         45328 function calls in 0.177 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.177    0.177 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.177    0.177 surrogate.py:1721(__call__)
        1    0.000    0.000    0.176    0.176 precessing_surrogate.py:1263(__call__)
        1    0.001    0.001    0.176    0.176 precessing_surrogate.py:933(__call__)
      218    0.001    0.000    0.133    0.001 surrogate.py:85(_splinterp_Cwrapper)
      218    0.017    0.000    0.132    0.001 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.129    0.026 precessing_surrogate.py:849(splinterp_many)
      725    0.056    0.000    0.056    0.000 {built-in method builtins.min}
      726    0.055    0.000    0.055    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:387(__call__)
      546    0.002    0.000    0.020    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:621(_integrate_forward)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:105(rotateWaveform)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:655(_integrate_backward)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
     2610    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
      872    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
      654    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      872    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      224    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:340(_get_t_from_omega)
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      872    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         45329 function calls in 0.128 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.128    0.128 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.128    0.128 surrogate.py:1721(__call__)
        1    0.000    0.000    0.128    0.128 precessing_surrogate.py:1263(__call__)
        1    0.001    0.001    0.128    0.128 precessing_surrogate.py:933(__call__)
      218    0.000    0.000    0.087    0.000 surrogate.py:85(_splinterp_Cwrapper)
      218    0.012    0.000    0.086    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.082    0.016 precessing_surrogate.py:849(splinterp_many)
      725    0.036    0.000    0.036    0.000 {built-in method builtins.min}
      726    0.034    0.000    0.034    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:387(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:621(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:105(rotateWaveform)
     5135    0.003    0.000    0.013    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
     5135    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:655(_integrate_backward)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
     2610    0.002    0.000    0.002    0.000 {built-in method numpy.array}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
      872    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      872    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      654    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:340(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      224    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      872    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         36878 function calls in 0.447 seconds

   Ordered by: cumulative time
   List reduced from 86 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.447    0.447 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.447    0.447 surrogate.py:1721(__call__)
        1    0.000    0.000    0.447    0.447 precessing_surrogate.py:1263(__call__)
        1    0.006    0.006    0.447    0.447 precessing_surrogate.py:933(__call__)
        5    0.000    0.000    0.399    0.080 precessing_surrogate.py:849(splinterp_many)
       75    0.000    0.000    0.396    0.005 surrogate.py:85(_splinterp_Cwrapper)
       75    0.040    0.001    0.395    0.005 spline_interp_Cwrapper.py:39(interpolate)
      426    0.176    0.000    0.176    0.000 {built-in method builtins.min}
      427    0.173    0.000    0.173    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:387(__call__)
        1    0.001    0.001    0.020    0.020 precessing_surrogate.py:621(_integrate_forward)
      505    0.001    0.000    0.019    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.002    0.002    0.017    0.017 precessing_surrogate.py:105(rotateWaveform)
        1    0.012    0.012    0.015    0.015 precessing_surrogate.py:42(_wignerD_matrices)
     1515    0.003    0.000    0.015    0.000 precessing_surrogate.py:179(_eval_vector_fit)
     4637    0.003    0.000    0.012    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
     4637    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     2147    0.006    0.000    0.006    0.000 {built-in method numpy.array}
       81    0.004    0.000    0.004    0.000 {built-in method numpy.zeros}
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
      225    0.002    0.000    0.002    0.000 {method 'astype' of 'numpy.ndarray' objects}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      300    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      300    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:556(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      300    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         46684 function calls in 0.142 seconds

   Ordered by: cumulative time
   List reduced from 96 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.142    0.142 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.142    0.142 surrogate.py:1721(__call__)
        1    0.000    0.000    0.142    0.142 precessing_surrogate.py:1263(__call__)
        1    0.001    0.001    0.142    0.142 precessing_surrogate.py:933(__call__)
      218    0.000    0.000    0.098    0.000 surrogate.py:85(_splinterp_Cwrapper)
      218    0.013    0.000    0.098    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.093    0.019 precessing_surrogate.py:849(splinterp_many)
      725    0.040    0.000    0.040    0.000 {built-in method builtins.min}
      726    0.040    0.000    0.040    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:387(__call__)
      546    0.002    0.000    0.020    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:105(rotateWaveform)
     5274    0.004    0.000    0.014    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.010    0.010    0.013    0.013 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:655(_integrate_backward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:621(_integrate_forward)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
     5274    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.007    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
     2749    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      872    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:340(_get_t_from_omega)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:334(get_omega)
      872    0.002    0.000    0.002    0.000 __init__.py:613(cast)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      654    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      224    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      872    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         36878 function calls in 0.136 seconds

   Ordered by: cumulative time
   List reduced from 86 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.136    0.136 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.136    0.136 surrogate.py:1721(__call__)
        1    0.000    0.000    0.136    0.136 precessing_surrogate.py:1263(__call__)
        1    0.001    0.001    0.136    0.136 precessing_surrogate.py:933(__call__)
        5    0.000    0.000    0.097    0.019 precessing_surrogate.py:849(splinterp_many)
       75    0.000    0.000    0.097    0.001 surrogate.py:85(_splinterp_Cwrapper)
       75    0.012    0.000    0.097    0.001 spline_interp_Cwrapper.py:39(interpolate)
      426    0.041    0.000    0.041    0.000 {built-in method builtins.min}
      427    0.041    0.000    0.041    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:387(__call__)
        1    0.001    0.001    0.020    0.020 precessing_surrogate.py:621(_integrate_forward)
      505    0.001    0.000    0.019    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
     1515    0.003    0.000    0.015    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:105(rotateWaveform)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.003    0.000    0.012    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
     4637    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
     2147    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      225    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
       81    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
      300    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      300    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:556(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      300    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         46684 function calls in 0.085 seconds

   Ordered by: cumulative time
   List reduced from 96 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.084    0.084 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.084    0.084 surrogate.py:1721(__call__)
        1    0.000    0.000    0.084    0.084 precessing_surrogate.py:1263(__call__)
        1    0.000    0.000    0.084    0.084 precessing_surrogate.py:933(__call__)
      218    0.000    0.000    0.042    0.000 surrogate.py:85(_splinterp_Cwrapper)
      218    0.008    0.000    0.041    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.037    0.007 precessing_surrogate.py:849(splinterp_many)
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:387(__call__)
      546    0.002    0.000    0.020    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:179(_eval_vector_fit)
      725    0.015    0.000    0.015    0.000 {built-in method builtins.min}
      726    0.015    0.000    0.015    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:105(rotateWaveform)
     5274    0.003    0.000    0.014    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.010    0.010    0.013    0.013 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:655(_integrate_backward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:621(_integrate_forward)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
     5274    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.007    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:340(_get_t_from_omega)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:334(get_omega)
      872    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
     2749    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
      872    0.002    0.000    0.002    0.000 __init__.py:613(cast)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      654    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      872    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
      224    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         68904 function calls (68884 primitive calls) in 0.517 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.518    0.518 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.001    0.001    0.518    0.518 surrogate.py:1721(__call__)
        1    0.000    0.000    0.513    0.513 surrogate.py:923(__call__)
        1    0.042    0.042    0.429    0.429 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.005    0.000    0.385    0.032 surrogate.py:85(_splinterp_Cwrapper)
       22    0.039    0.002    0.380    0.017 spline_interp_Cwrapper.py:39(interpolate)
       44    0.167    0.004    0.167    0.004 {built-in method builtins.min}
       44    0.164    0.004    0.164    0.004 {built-in method builtins.max}
       11    0.000    0.000    0.083    0.008 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.083    0.008 surrogate.py:401(__call__)
       20    0.000    0.000    0.083    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.074    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.074    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.068    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.068    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.067    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.043    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
      316    0.004    0.000    0.035    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.002    0.000    0.027    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.004    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
      948    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
       24    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      224    0.008    0.000    0.008    0.000 {method 'astype' of 'numpy.ndarray' objects}
      158    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6164    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      316    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         68900 function calls (68880 primitive calls) in 0.191 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.191    0.191 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.191    0.191 surrogate.py:1721(__call__)
        1    0.000    0.000    0.191    0.191 surrogate.py:923(__call__)
        1    0.012    0.012    0.106    0.106 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.002    0.000    0.093    0.008 surrogate.py:85(_splinterp_Cwrapper)
       22    0.012    0.001    0.090    0.004 spline_interp_Cwrapper.py:39(interpolate)
       11    0.000    0.000    0.084    0.008 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.084    0.008 surrogate.py:401(__call__)
       20    0.000    0.000    0.084    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.075    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.075    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.069    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.068    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.043    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
       44    0.037    0.001    0.037    0.001 {built-in method builtins.min}
       44    0.036    0.001    0.036    0.001 {built-in method builtins.max}
      316    0.004    0.000    0.035    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.002    0.000    0.027    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.004    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
      948    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
       24    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      158    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
      224    0.005    0.000    0.005    0.000 {method 'astype' of 'numpy.ndarray' objects}
     6164    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      316    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      316    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         68904 function calls (68884 primitive calls) in 0.866 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.867    0.867 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.004    0.004    0.867    0.867 surrogate.py:1721(__call__)
        1    0.000    0.000    0.857    0.857 surrogate.py:923(__call__)
        1    0.078    0.078    0.774    0.774 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.015    0.000    0.693    0.058 surrogate.py:85(_splinterp_Cwrapper)
       22    0.070    0.003    0.678    0.031 spline_interp_Cwrapper.py:39(interpolate)
       44    0.302    0.007    0.302    0.007 {built-in method builtins.min}
       44    0.295    0.007    0.295    0.007 {built-in method builtins.max}
       11    0.000    0.000    0.082    0.007 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.082    0.007 surrogate.py:401(__call__)
       20    0.000    0.000    0.082    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.074    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.073    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.067    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.067    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.067    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.043    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
      316    0.004    0.000    0.035    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.026    0.000 _base.py:297(predict)
      158    0.002    0.000    0.026    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.004    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
      948    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
      224    0.009    0.000    0.009    0.000 {method 'astype' of 'numpy.ndarray' objects}
       24    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      158    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
        1    0.000    0.000    0.005    0.005 {method 'update' of 'dict' objects}
       21    0.005    0.000    0.005    0.000 surrogate.py:2126(<genexpr>)
     6164    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.001    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      316    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         68900 function calls (68880 primitive calls) in 0.206 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.207    0.207 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.207    0.207 surrogate.py:1721(__call__)
        1    0.000    0.000    0.206    0.206 surrogate.py:923(__call__)
        1    0.013    0.013    0.124    0.124 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.001    0.000    0.111    0.009 surrogate.py:85(_splinterp_Cwrapper)
       22    0.014    0.001    0.109    0.005 spline_interp_Cwrapper.py:39(interpolate)
       11    0.000    0.000    0.081    0.007 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.081    0.007 surrogate.py:401(__call__)
       20    0.000    0.000    0.081    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.073    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.073    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.067    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.067    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.067    0.000 evaluate_fit.py:128(GPR_predict)
       44    0.046    0.001    0.046    0.001 {built-in method builtins.min}
       44    0.046    0.001    0.046    0.001 {built-in method builtins.max}
      316    0.002    0.000    0.042    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
      316    0.004    0.000    0.035    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.026    0.000 _base.py:297(predict)
      158    0.001    0.000    0.026    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.004    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
      948    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
       24    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      158    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6164    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      316    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      948    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         68877 function calls (68857 primitive calls) in 0.222 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.222    0.222 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.222    0.222 surrogate.py:1721(__call__)
        1    0.000    0.000    0.222    0.222 surrogate.py:923(__call__)
        1    0.014    0.014    0.139    0.139 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.001    0.000    0.123    0.010 surrogate.py:85(_splinterp_Cwrapper)
       22    0.017    0.001    0.122    0.006 spline_interp_Cwrapper.py:39(interpolate)
       11    0.000    0.000    0.082    0.007 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.082    0.007 surrogate.py:401(__call__)
       20    0.000    0.000    0.082    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.074    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.074    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.068    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.068    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.067    0.000 evaluate_fit.py:128(GPR_predict)
       44    0.051    0.001    0.051    0.001 {built-in method builtins.min}
       44    0.050    0.001    0.050    0.001 {built-in method builtins.max}
      316    0.002    0.000    0.043    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
      316    0.004    0.000    0.035    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.026    0.000 _base.py:297(predict)
      158    0.002    0.000    0.026    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.004    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
      948    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
       24    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      158    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6164    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      316    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      224    0.003    0.000    0.003    0.000 {method 'astype' of 'numpy.ndarray' objects}
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         68877 function calls (68857 primitive calls) in 4.101 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    4.102    4.102 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.023    0.023    4.102    4.102 surrogate.py:1721(__call__)
        1    0.000    0.000    4.066    4.066 surrogate.py:923(__call__)
        1    0.434    0.434    3.984    3.984 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.042    0.001    3.544    0.295 surrogate.py:85(_splinterp_Cwrapper)
       22    0.329    0.015    3.501    0.159 spline_interp_Cwrapper.py:39(interpolate)
       44    1.593    0.036    1.593    0.036 {built-in method builtins.min}
       44    1.540    0.035    1.540    0.035 {built-in method builtins.max}
       11    0.000    0.000    0.082    0.007 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.082    0.007 surrogate.py:401(__call__)
       20    0.000    0.000    0.082    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.074    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.073    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.067    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.067    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.067    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.042    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
      316    0.004    0.000    0.035    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.026    0.000 _base.py:297(predict)
      158    0.002    0.000    0.026    0.000 _base.py:287(_decision_function)
      224    0.025    0.000    0.025    0.000 {method 'astype' of 'numpy.ndarray' objects}
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      180    0.014    0.000    0.014    0.000 {built-in method numpy.zeros}
      316    0.004    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
        9    0.013    0.001    0.013    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
      948    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
       24    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      158    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6164    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      316    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         68877 function calls (68857 primitive calls) in 0.169 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.170    0.170 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.170    0.170 surrogate.py:1721(__call__)
        1    0.000    0.000    0.169    0.169 surrogate.py:923(__call__)
       11    0.000    0.000    0.085    0.008 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.085    0.008 surrogate.py:401(__call__)
       20    0.000    0.000    0.085    0.004 surrogate.py:276(__call__)
        1    0.009    0.009    0.084    0.084 surrogate.py:726(_coorbital_to_inertial_frame)
      158    0.000    0.000    0.076    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.075    0.000 nodeFunction.py:110(__call__)
    32/12    0.001    0.000    0.074    0.006 surrogate.py:85(_splinterp_Cwrapper)
       22    0.010    0.000    0.072    0.003 spline_interp_Cwrapper.py:39(interpolate)
      158    0.000    0.000    0.069    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.069    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.043    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
      316    0.004    0.000    0.036    0.000 validation.py:725(check_array)
       44    0.030    0.001    0.030    0.001 {built-in method builtins.min}
       44    0.030    0.001    0.030    0.001 {built-in method builtins.max}
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.002    0.000    0.027    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.004    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
      948    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
       24    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      158    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6164    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      158    0.001    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.000    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      316    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      948    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         68877 function calls (68857 primitive calls) in 0.976 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.977    0.977 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.005    0.005    0.977    0.977 surrogate.py:1721(__call__)
        1    0.000    0.000    0.970    0.970 surrogate.py:923(__call__)
        1    0.090    0.090    0.887    0.887 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.016    0.001    0.795    0.066 surrogate.py:85(_splinterp_Cwrapper)
       22    0.078    0.004    0.778    0.035 spline_interp_Cwrapper.py:39(interpolate)
       44    0.347    0.008    0.347    0.008 {built-in method builtins.min}
       44    0.342    0.008    0.342    0.008 {built-in method builtins.max}
       11    0.000    0.000    0.082    0.007 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.082    0.007 surrogate.py:401(__call__)
       20    0.000    0.000    0.082    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.073    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.073    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.067    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.067    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.067    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.042    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
      316    0.004    0.000    0.035    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.026    0.000 _base.py:297(predict)
      158    0.002    0.000    0.026    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.004    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
      948    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
       24    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      224    0.008    0.000    0.008    0.000 {method 'astype' of 'numpy.ndarray' objects}
      158    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6164    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      316    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      180    0.003    0.000    0.003    0.000 {built-in method numpy.zeros}
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         22024 function calls (22016 primitive calls) in 0.197 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.198    0.198 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.001    0.001    0.198    0.198 surrogate.py:1721(__call__)
        1    0.000    0.000    0.195    0.195 surrogate.py:923(__call__)
        1    0.019    0.019    0.175    0.175 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.001    0.000    0.155    0.026 surrogate.py:85(_splinterp_Cwrapper)
       10    0.018    0.002    0.154    0.015 spline_interp_Cwrapper.py:39(interpolate)
       20    0.068    0.003    0.068    0.003 {built-in method builtins.max}
       20    0.066    0.003    0.066    0.003 {built-in method builtins.min}
        5    0.000    0.000    0.020    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.020    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.020    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.019    0.000 nodeFunction.py:205(__call__)
       50    0.001    0.000    0.019    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.017    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.017    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.001    0.000    0.011    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.010    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.007    0.000 _base.py:297(predict)
       50    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       80    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
       11    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         22020 function calls (22012 primitive calls) in 0.049 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.049    0.049 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.049    0.049 surrogate.py:1721(__call__)
        1    0.000    0.000    0.048    0.048 surrogate.py:923(__call__)
        1    0.003    0.003    0.028    0.028 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.000    0.000    0.025    0.004 surrogate.py:85(_splinterp_Cwrapper)
       10    0.004    0.000    0.024    0.002 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.020    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.020    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.020    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.019    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.019    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.017    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.017    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.001    0.000    0.011    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.010    0.000 _gpr.py:373(predict)
       20    0.010    0.000    0.010    0.000 {built-in method builtins.max}
       20    0.009    0.000    0.009    0.000 {built-in method builtins.min}
      100    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.007    0.000 _base.py:297(predict)
       50    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       80    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
       50    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
      150    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         22024 function calls (22016 primitive calls) in 0.358 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.359    0.359 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.001    0.001    0.359    0.359 surrogate.py:1721(__call__)
        1    0.000    0.000    0.354    0.354 surrogate.py:923(__call__)
        1    0.035    0.035    0.334    0.334 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.003    0.000    0.297    0.049 surrogate.py:85(_splinterp_Cwrapper)
       10    0.028    0.003    0.294    0.029 spline_interp_Cwrapper.py:39(interpolate)
       20    0.132    0.007    0.132    0.007 {built-in method builtins.max}
       20    0.127    0.006    0.127    0.006 {built-in method builtins.min}
        5    0.000    0.000    0.020    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.020    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.020    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.019    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.019    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.017    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.017    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.011    0.000 validation.py:2793(validate_data)
       50    0.001    0.000    0.010    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.007    0.000 _base.py:297(predict)
       50    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       80    0.005    0.000    0.005    0.000 {method 'astype' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       60    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         22020 function calls (22012 primitive calls) in 0.060 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.060    0.060 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.060    0.060 surrogate.py:1721(__call__)
        1    0.000    0.000    0.059    0.059 surrogate.py:923(__call__)
        1    0.004    0.004    0.038    0.038 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.001    0.000    0.033    0.006 surrogate.py:85(_splinterp_Cwrapper)
       10    0.005    0.000    0.032    0.003 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.021    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.021    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.021    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.019    0.000 nodeFunction.py:205(__call__)
       50    0.001    0.000    0.019    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.018    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.018    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.018    0.000 evaluate_fit.py:128(GPR_predict)
       20    0.013    0.001    0.013    0.001 {built-in method builtins.max}
       20    0.013    0.001    0.013    0.001 {built-in method builtins.min}
      100    0.001    0.000    0.011    0.000 validation.py:2793(validate_data)
       50    0.001    0.000    0.010    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.007    0.000 _base.py:297(predict)
       50    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       80    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
     2950    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         22007 function calls (21999 primitive calls) in 0.063 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.063    0.063 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.063    0.063 surrogate.py:1721(__call__)
        1    0.000    0.000    0.063    0.063 surrogate.py:923(__call__)
        1    0.004    0.004    0.043    0.043 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.000    0.000    0.037    0.006 surrogate.py:85(_splinterp_Cwrapper)
       10    0.006    0.001    0.037    0.004 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.020    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.020    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.020    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.019    0.000 nodeFunction.py:205(__call__)
       50    0.001    0.000    0.019    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.017    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.017    0.000 evaluate_fit.py:128(GPR_predict)
       20    0.015    0.001    0.015    0.001 {built-in method builtins.max}
       20    0.015    0.001    0.015    0.001 {built-in method builtins.min}
      100    0.001    0.000    0.011    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.010    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.007    0.000 _base.py:297(predict)
       50    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       80    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
     2950    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         22007 function calls (21999 primitive calls) in 1.833 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.833    1.833 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.014    0.014    1.833    1.833 surrogate.py:1721(__call__)
        1    0.000    0.000    1.811    1.811 surrogate.py:923(__call__)
        1    0.202    0.202    1.790    1.790 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.015    0.001    1.583    0.264 surrogate.py:85(_splinterp_Cwrapper)
       10    0.144    0.014    1.568    0.157 spline_interp_Cwrapper.py:39(interpolate)
       20    0.705    0.035    0.705    0.035 {built-in method builtins.min}
       20    0.701    0.035    0.701    0.035 {built-in method builtins.max}
        5    0.000    0.000    0.021    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.021    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.021    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.019    0.000 nodeFunction.py:205(__call__)
       50    0.001    0.000    0.019    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.018    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.018    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.018    0.000 evaluate_fit.py:128(GPR_predict)
       80    0.011    0.000    0.011    0.000 {method 'astype' of 'numpy.ndarray' objects}
      100    0.001    0.000    0.011    0.000 validation.py:2793(validate_data)
       50    0.001    0.000    0.010    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.009    0.000 validation.py:725(check_array)
        5    0.008    0.002    0.008    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.007    0.000 _base.py:297(predict)
       50    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       60    0.006    0.000    0.006    0.000 {built-in method numpy.zeros}
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1402(diff)
        2    0.001    0.001    0.001    0.001 surrogate.py:716(_search_omega)
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         22007 function calls (21999 primitive calls) in 0.043 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.043    0.043 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.043    0.043 surrogate.py:1721(__call__)
        1    0.000    0.000    0.043    0.043 surrogate.py:923(__call__)
        1    0.002    0.002    0.023    0.023 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.000    0.000    0.020    0.003 surrogate.py:85(_splinterp_Cwrapper)
        5    0.000    0.000    0.020    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.020    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.020    0.002 surrogate.py:276(__call__)
       10    0.004    0.000    0.020    0.002 spline_interp_Cwrapper.py:39(interpolate)
       50    0.000    0.000    0.019    0.000 nodeFunction.py:205(__call__)
       50    0.001    0.000    0.019    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.017    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.017    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.001    0.000    0.011    0.000 validation.py:2793(validate_data)
       50    0.001    0.000    0.010    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       20    0.008    0.000    0.008    0.000 {built-in method builtins.max}
       20    0.008    0.000    0.008    0.000 {built-in method builtins.min}
       50    0.000    0.000    0.007    0.000 _base.py:297(predict)
       50    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       80    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      150    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
     2950    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         22007 function calls (21999 primitive calls) in 0.413 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.414    0.414 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.002    0.002    0.414    0.414 surrogate.py:1721(__call__)
        1    0.000    0.000    0.410    0.410 surrogate.py:923(__call__)
        1    0.041    0.041    0.390    0.390 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.004    0.000    0.347    0.058 surrogate.py:85(_splinterp_Cwrapper)
       10    0.039    0.004    0.343    0.034 spline_interp_Cwrapper.py:39(interpolate)
       20    0.150    0.008    0.150    0.008 {built-in method builtins.max}
       20    0.149    0.007    0.149    0.007 {built-in method builtins.min}
        5    0.000    0.000    0.020    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.020    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.020    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.019    0.000 nodeFunction.py:205(__call__)
       50    0.001    0.000    0.019    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.017    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.017    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.001    0.000    0.011    0.000 validation.py:2793(validate_data)
       50    0.001    0.000    0.010    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.007    0.000 _base.py:297(predict)
       50    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
       80    0.004    0.000    0.004    0.000 {method 'astype' of 'numpy.ndarray' objects}
      100    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       60    0.002    0.000    0.002    0.000 {built-in method numpy.zeros}
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         81672 function calls (81654 primitive calls) in 0.491 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.492    0.492 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.001    0.001    0.492    0.492 surrogate.py:1721(__call__)
        1    0.000    0.000    0.487    0.487 surrogate.py:923(__call__)
        1    0.037    0.037    0.390    0.390 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.003    0.000    0.351    0.032 surrogate.py:85(_splinterp_Cwrapper)
       20    0.035    0.002    0.348    0.017 spline_interp_Cwrapper.py:39(interpolate)
       40    0.154    0.004    0.154    0.004 {built-in method builtins.max}
       40    0.152    0.004    0.152    0.004 {built-in method builtins.min}
       10    0.000    0.000    0.097    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.097    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.097    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.087    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.087    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.080    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.080    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.079    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.050    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.041    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.031    0.000 _base.py:297(predict)
      188    0.002    0.000    0.031    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.005    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       21    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      248    0.005    0.000    0.005    0.000 {method 'astype' of 'numpy.ndarray' objects}
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      376    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    11092    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      564    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         81668 function calls (81650 primitive calls) in 0.187 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.188    0.188 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.188    0.188 surrogate.py:1721(__call__)
        1    0.000    0.000    0.188    0.188 surrogate.py:923(__call__)
       10    0.000    0.000    0.098    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.098    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.098    0.006 surrogate.py:276(__call__)
        1    0.009    0.009    0.089    0.089 surrogate.py:726(_coorbital_to_inertial_frame)
      188    0.000    0.000    0.089    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.089    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.082    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.081    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.081    0.000 evaluate_fit.py:128(GPR_predict)
    29/11    0.001    0.000    0.080    0.007 surrogate.py:85(_splinterp_Cwrapper)
       20    0.010    0.000    0.078    0.004 spline_interp_Cwrapper.py:39(interpolate)
      376    0.002    0.000    0.051    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.047    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.042    0.000 validation.py:725(check_array)
       40    0.033    0.001    0.033    0.001 {built-in method builtins.min}
       40    0.033    0.001    0.033    0.001 {built-in method builtins.max}
      188    0.000    0.000    0.032    0.000 _base.py:297(predict)
      188    0.002    0.000    0.032    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.005    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.012    0.000 validation.py:371(_num_samples)
       21    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      376    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      376    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         81672 function calls (81654 primitive calls) in 0.787 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.788    0.788 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.002    0.002    0.788    0.788 surrogate.py:1721(__call__)
        1    0.000    0.000    0.779    0.779 surrogate.py:923(__call__)
        1    0.065    0.065    0.682    0.682 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.005    0.000    0.615    0.056 surrogate.py:85(_splinterp_Cwrapper)
       20    0.060    0.003    0.610    0.031 spline_interp_Cwrapper.py:39(interpolate)
       40    0.274    0.007    0.274    0.007 {built-in method builtins.max}
       40    0.268    0.007    0.268    0.007 {built-in method builtins.min}
       10    0.000    0.000    0.096    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.096    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.096    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.087    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.087    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.080    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.080    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.079    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.050    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.041    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.031    0.000 _base.py:297(predict)
      188    0.002    0.000    0.031    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       21    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      248    0.005    0.000    0.005    0.000 {method 'astype' of 'numpy.ndarray' objects}
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
        1    0.000    0.000    0.005    0.005 {method 'update' of 'dict' objects}
       18    0.005    0.000    0.005    0.000 surrogate.py:2126(<genexpr>)
      564    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      376    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         81668 function calls (81650 primitive calls) in 0.218 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.218    0.218 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.218    0.218 surrogate.py:1721(__call__)
        1    0.000    0.000    0.218    0.218 surrogate.py:923(__call__)
        1    0.013    0.013    0.119    0.119 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.002    0.000    0.105    0.010 surrogate.py:85(_splinterp_Cwrapper)
       20    0.012    0.001    0.103    0.005 spline_interp_Cwrapper.py:39(interpolate)
       10    0.000    0.000    0.098    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.098    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.098    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.087    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.087    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.080    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.080    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.080    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.050    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
       40    0.044    0.001    0.044    0.001 {built-in method builtins.min}
       40    0.043    0.001    0.043    0.001 {built-in method builtins.max}
      376    0.005    0.000    0.041    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.031    0.000 _base.py:297(predict)
      188    0.002    0.000    0.031    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.005    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      376    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    11092    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      564    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      248    0.003    0.000    0.003    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         81648 function calls (81630 primitive calls) in 0.227 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.228    0.228 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.228    0.228 surrogate.py:1721(__call__)
        1    0.000    0.000    0.227    0.227 surrogate.py:923(__call__)
        1    0.013    0.013    0.129    0.129 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.002    0.000    0.115    0.010 surrogate.py:85(_splinterp_Cwrapper)
       20    0.015    0.001    0.113    0.006 spline_interp_Cwrapper.py:39(interpolate)
       10    0.000    0.000    0.098    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.098    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.097    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.087    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.087    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.080    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.080    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.079    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.050    0.000 validation.py:2793(validate_data)
       40    0.048    0.001    0.048    0.001 {built-in method builtins.min}
       40    0.047    0.001    0.047    0.001 {built-in method builtins.max}
      188    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.041    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.031    0.000 _base.py:297(predict)
      188    0.002    0.000    0.031    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.005    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      376    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      564    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
    11092    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
     1128    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         81648 function calls (81630 primitive calls) in 3.735 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    3.736    3.736 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.023    0.023    3.736    3.736 surrogate.py:1721(__call__)
        1    0.000    0.000    3.701    3.701 surrogate.py:923(__call__)
        1    0.373    0.373    3.604    3.604 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.038    0.001    3.225    0.293 surrogate.py:85(_splinterp_Cwrapper)
       20    0.297    0.015    3.187    0.159 spline_interp_Cwrapper.py:39(interpolate)
       40    1.444    0.036    1.444    0.036 {built-in method builtins.min}
       40    1.409    0.035    1.409    0.035 {built-in method builtins.max}
       10    0.000    0.000    0.096    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.096    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.096    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.087    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.087    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.080    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.079    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.079    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.050    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.041    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.031    0.000 _base.py:297(predict)
      188    0.002    0.000    0.031    0.000 _base.py:287(_decision_function)
      248    0.024    0.000    0.024    0.000 {method 'astype' of 'numpy.ndarray' objects}
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
        7    0.012    0.002    0.012    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      208    0.012    0.000    0.012    0.000 {built-in method numpy.zeros}
     1128    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       21    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      376    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         81648 function calls (81630 primitive calls) in 0.175 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.176    0.176 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.176    0.176 surrogate.py:1721(__call__)
        1    0.000    0.000    0.175    0.175 surrogate.py:923(__call__)
       10    0.000    0.000    0.100    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.100    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.100    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.089    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.089    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.082    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.082    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.081    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.008    0.008    0.075    0.075 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.001    0.000    0.066    0.006 surrogate.py:85(_splinterp_Cwrapper)
       20    0.008    0.000    0.064    0.003 spline_interp_Cwrapper.py:39(interpolate)
      376    0.002    0.000    0.051    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.047    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.042    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.032    0.000 _base.py:297(predict)
      188    0.002    0.000    0.032    0.000 _base.py:287(_decision_function)
       40    0.027    0.001    0.027    0.001 {built-in method builtins.min}
       40    0.027    0.001    0.027    0.001 {built-in method builtins.max}
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.005    0.000    0.017    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.012    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      376    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
     1128    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         81648 function calls (81630 primitive calls) in 0.898 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.899    0.899 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.006    0.006    0.899    0.899 surrogate.py:1721(__call__)
        1    0.000    0.000    0.892    0.892 surrogate.py:923(__call__)
        1    0.077    0.077    0.795    0.795 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.007    0.000    0.715    0.065 surrogate.py:85(_splinterp_Cwrapper)
       20    0.070    0.004    0.709    0.035 spline_interp_Cwrapper.py:39(interpolate)
       40    0.316    0.008    0.316    0.008 {built-in method builtins.min}
       40    0.313    0.008    0.313    0.008 {built-in method builtins.max}
       10    0.000    0.000    0.097    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.097    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.096    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.087    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.087    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.080    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.080    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.080    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.051    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.041    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.031    0.000 _base.py:297(predict)
      188    0.002    0.000    0.031    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       21    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
      248    0.006    0.000    0.006    0.000 {method 'astype' of 'numpy.ndarray' objects}
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      376    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
    11092    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
```

#### PR-71

##### NRSur7dq4 / mks_dt_0.000244140625_flow_0

```text
         27100 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1721(__call__)
        1    0.000    0.000    0.032    0.032 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:110(rotateWaveform)
        1    0.006    0.006    0.007    0.007 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      837    0.002    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1633    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         27101 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1721(__call__)
        1    0.000    0.000    0.032    0.032 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:847(inertial_waveform_modes)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:110(rotateWaveform)
        1    0.006    0.006    0.008    0.008 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:660(_integrate_backward)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     1633    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.000    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         27100 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1721(__call__)
        1    0.000    0.000    0.034    0.034 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:110(rotateWaveform)
        1    0.007    0.007    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:626(_integrate_forward)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
     1633    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.000    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         27101 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1721(__call__)
        1    0.000    0.000    0.034    0.034 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:110(rotateWaveform)
        1    0.007    0.007    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:769(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
     1633    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.000    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         20326 function calls in 0.027 seconds

   Ordered by: cumulative time
   List reduced from 91 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.027    0.027 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.027    0.027 surrogate.py:1721(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:626(_integrate_forward)
      238    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        4    0.000    0.000    0.006    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.005    0.005 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.005    0.005    0.005    0.005 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2390    0.001    0.000    0.004    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
     2390    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     1252    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         27707 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_forward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2876    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      644    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1703    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.000    0.000 precessing_surrogate.py:854(splinterp_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         20326 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 91 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:626(_integrate_forward)
      238    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2390    0.001    0.000    0.004    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
     2390    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     1252    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         27707 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_forward)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2876    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      644    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     1703    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
      644    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         43096 function calls in 0.048 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.048    0.048 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.048    0.048 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         43097 function calls in 0.047 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.002    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     2605    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         43096 function calls in 0.048 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.048    0.048 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.048    0.048 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         43097 function calls in 0.047 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         35075 function calls in 0.046 seconds

   Ordered by: cumulative time
   List reduced from 94 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.046    0.046 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.046    0.046 surrogate.py:1721(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.019    0.019 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.018    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.003    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.003    0.000    0.012    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        4    0.000    0.000    0.010    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.009    0.009 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.009    0.009 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     4637    0.001    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     2142    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         44452 function calls in 0.048 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.048    0.048 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.048    0.048 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:392(__call__)
      546    0.002    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     5274    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:660(_integrate_backward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:626(_integrate_forward)
     5274    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:345(_get_t_from_omega)
     2744    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:339(get_omega)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         35075 function calls in 0.040 seconds

   Ordered by: cumulative time
   List reduced from 94 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.040    0.040 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.040    0.040 surrogate.py:1721(__call__)
        1    0.000    0.000    0.040    0.040 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.040    0.040 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.019    0.019 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.018    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.003    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.003    0.000    0.011    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
     4637    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.001    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
     2142    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         44452 function calls in 0.047 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:392(__call__)
      546    0.002    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
     5274    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.010    0.010    0.013    0.013 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:660(_integrate_backward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:626(_integrate_forward)
     5274    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:345(_get_t_from_omega)
     2744    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:339(get_omega)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         76481 function calls in 0.140 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.141    0.141 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.001    0.001    0.141    0.141 surrogate.py:1721(__call__)
        1    0.005    0.005    0.134    0.134 surrogate.py:934(__call__)
       12    0.000    0.000    0.096    0.008 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.096    0.008 surrogate.py:417(__call__)
       22    0.000    0.000    0.096    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.085    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.078    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.078    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      354    0.005    0.000    0.040    0.000 validation.py:725(check_array)
        1    0.001    0.001    0.033    0.033 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.030    0.000 _base.py:297(predict)
      177    0.002    0.000    0.030    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      354    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
        1    0.000    0.000    0.015    0.015 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.014    0.014    0.015    0.015 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       26    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
     6905    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
      354    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      177    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      531    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       21    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      531    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      354    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      708    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         76477 function calls in 0.105 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.106    0.106 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.106    0.106 surrogate.py:1721(__call__)
        1    0.002    0.002    0.105    0.105 surrogate.py:934(__call__)
       12    0.000    0.000    0.095    0.008 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.095    0.008 surrogate.py:417(__call__)
       22    0.000    0.000    0.095    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.085    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.078    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.078    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.049    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.045    0.000 _gpr.py:373(predict)
      354    0.005    0.000    0.040    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.030    0.000 _base.py:297(predict)
      177    0.002    0.000    0.030    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      354    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       26    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
     6905    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      354    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      531    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      177    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      531    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      531    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      354    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      354    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      354    0.000    0.000    0.003    0.000 _base.py:711(__sklearn_tags__)
      354    0.001    0.000    0.003    0.000 fromnumeric.py:66(_wrapreduction)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         76481 function calls in 0.180 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.181    0.181 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.002    0.002    0.181    0.181 surrogate.py:1721(__call__)
        1    0.004    0.004    0.159    0.159 surrogate.py:934(__call__)
       12    0.000    0.000    0.096    0.008 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.096    0.008 surrogate.py:417(__call__)
       22    0.000    0.000    0.096    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.085    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.078    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.003    0.003    0.058    0.058 surrogate.py:742(_coorbital_to_inertial_frame)
      354    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.045    0.000 _gpr.py:373(predict)
      354    0.005    0.000    0.040    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.030    0.000 _base.py:297(predict)
      177    0.002    0.000    0.030    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.023    0.023    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.017    0.017 {method 'update' of 'dict' objects}
       21    0.017    0.001    0.017    0.001 surrogate.py:2126(<genexpr>)
      354    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
       26    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
     6905    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      354    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      177    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      531    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      177    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      531    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      354    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      708    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         76477 function calls in 0.105 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.106    0.106 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.106    0.106 surrogate.py:1721(__call__)
        1    0.002    0.002    0.104    0.104 surrogate.py:934(__call__)
       12    0.000    0.000    0.093    0.008 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.093    0.008 surrogate.py:417(__call__)
       22    0.000    0.000    0.093    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.084    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.084    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.077    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.077    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.045    0.000 _gpr.py:373(predict)
      354    0.005    0.000    0.040    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.030    0.000 _base.py:297(predict)
      177    0.002    0.000    0.030    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      354    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      177    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
     6905    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      354    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      177    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      531    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      531    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      354    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      708    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      354    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         76454 function calls in 0.104 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.105    0.105 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.105    0.105 surrogate.py:1721(__call__)
        1    0.002    0.002    0.105    0.105 surrogate.py:934(__call__)
       12    0.000    0.000    0.093    0.008 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.093    0.008 surrogate.py:417(__call__)
       22    0.000    0.000    0.093    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.084    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.084    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.077    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.077    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.045    0.000 _gpr.py:373(predict)
      354    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.030    0.000 _base.py:297(predict)
      177    0.002    0.000    0.030    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      354    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
     6905    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      531    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      177    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      354    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      177    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      531    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      531    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      354    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      354    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         76454 function calls in 0.398 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.398    0.398 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.034    0.034    0.398    0.398 surrogate.py:1721(__call__)
        1    0.003    0.003    0.345    0.345 surrogate.py:934(__call__)
        1    0.011    0.011    0.248    0.248 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.103    0.103 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.103    0.103    0.103    0.103 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.092    0.008 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.092    0.008 surrogate.py:417(__call__)
       22    0.000    0.000    0.092    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.083    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.083    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.076    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.076    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.076    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.073    0.073    0.073    0.073 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.053    0.027 surrogate.py:91(_splinterp_Cwrapper)
        2    0.052    0.026    0.053    0.027 spline_interp_Cwrapper.py:50(interpolate)
      354    0.002    0.000    0.047    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.044    0.000 _gpr.py:373(predict)
      354    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.030    0.000 _base.py:297(predict)
      177    0.002    0.000    0.029    0.000 _base.py:287(_decision_function)
        9    0.019    0.002    0.019    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.018    0.000 kernels.py:833(__call__)
      354    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
     6905    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      354    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      531    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      177    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      531    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      531    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
      354    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         76454 function calls in 0.105 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.106    0.106 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.106    0.106 surrogate.py:1721(__call__)
        1    0.002    0.002    0.105    0.105 surrogate.py:934(__call__)
       12    0.000    0.000    0.095    0.008 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.095    0.008 surrogate.py:417(__call__)
       22    0.000    0.000    0.095    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.085    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.078    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.049    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.045    0.000 _gpr.py:373(predict)
      354    0.005    0.000    0.040    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.030    0.000 _base.py:297(predict)
      177    0.002    0.000    0.030    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      354    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       26    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
     6905    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      354    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      531    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      177    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      531    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      531    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      354    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
    10443    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     1416    0.001    0.000    0.003    0.000 _config.py:35(get_config)
      354    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      354    0.001    0.000    0.003    0.000 fromnumeric.py:66(_wrapreduction)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         76454 function calls in 0.167 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.168    0.168 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.003    0.003    0.168    0.168 surrogate.py:1721(__call__)
        1    0.003    0.003    0.162    0.162 surrogate.py:934(__call__)
       12    0.000    0.000    0.096    0.008 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.096    0.008 surrogate.py:417(__call__)
       22    0.000    0.000    0.095    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.085    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.078    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.002    0.002    0.062    0.062 surrogate.py:742(_coorbital_to_inertial_frame)
      354    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.045    0.000 _gpr.py:373(predict)
      354    0.005    0.000    0.040    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.030    0.000 _base.py:297(predict)
      177    0.002    0.000    0.030    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.027    0.027 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.027    0.027    0.027    0.027 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.020    0.020    0.020    0.020 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      354    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
       26    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
     6905    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      354    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      177    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      531    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      531    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      354    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
      708    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         28686 function calls in 0.045 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.045    0.045 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.001    0.001    0.045    0.045 surrogate.py:1721(__call__)
        1    0.000    0.000    0.042    0.042 surrogate.py:934(__call__)
        6    0.000    0.000    0.026    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.026    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.026    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.025    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.025    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.023    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.023    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.023    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.001    0.001    0.015    0.015 surrogate.py:742(_coorbital_to_inertial_frame)
      132    0.001    0.000    0.014    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.013    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.012    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.009    0.000 _base.py:297(predict)
       66    0.001    0.000    0.009    0.000 _base.py:287(_decision_function)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
        1    0.005    0.005    0.005    0.005 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      132    0.001    0.000    0.005    0.000 validation.py:103(_assert_all_finite)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.004    0.004    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       66    0.000    0.000    0.004    0.000 kernels.py:931(__call__)
        2    0.000    0.000    0.004    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.004    0.002 spline_interp_Cwrapper.py:50(interpolate)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.002    0.000 {built-in method builtins.isinstance}
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
       11    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      198    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         28682 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.030    0.030 surrogate.py:934(__call__)
        6    0.000    0.000    0.026    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.026    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.026    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.024    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.024    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.022    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.022    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.022    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.014    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.013    0.000 _gpr.py:373(predict)
      132    0.002    0.000    0.012    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.009    0.000 _base.py:297(predict)
       66    0.001    0.000    0.009    0.000 _base.py:287(_decision_function)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.005    0.000 validation.py:103(_assert_all_finite)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.004    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      198    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      198    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
      132    0.000    0.000    0.001    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         28686 function calls in 0.058 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.059    0.059 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.001    0.001    0.059    0.059 surrogate.py:1721(__call__)
        1    0.000    0.000    0.054    0.054 surrogate.py:934(__call__)
        6    0.000    0.000    0.027    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.027    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.027    0.002 surrogate.py:292(__call__)
        1    0.001    0.001    0.026    0.026 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.025    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.025    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.023    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.023    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.023    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.014    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.013    0.000 _gpr.py:373(predict)
      132    0.002    0.000    0.012    0.000 validation.py:725(check_array)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.009    0.000 _base.py:297(predict)
       66    0.001    0.000    0.009    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.007    0.007 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.007    0.007    0.007    0.007 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.007    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.006    0.003    0.007    0.003 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.005    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.004    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         28682 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 surrogate.py:934(__call__)
        6    0.000    0.000    0.026    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.026    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.025    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.024    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.024    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.022    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.022    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.022    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.014    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.013    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.012    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.009    0.000 _base.py:297(predict)
       66    0.000    0.000    0.009    0.000 _base.py:287(_decision_function)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
        1    0.000    0.000    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
      132    0.001    0.000    0.005    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.004    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.002    0.000 {built-in method builtins.isinstance}
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      198    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      198    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
     3894    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         28669 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1721(__call__)
        1    0.000    0.000    0.032    0.032 surrogate.py:934(__call__)
        6    0.000    0.000    0.026    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.026    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.026    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.025    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.024    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.022    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.022    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.022    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.014    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.013    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.012    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.009    0.000 _base.py:297(predict)
       66    0.001    0.000    0.009    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.005    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.004    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      198    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      198    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
     3894    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         28669 function calls in 0.191 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.191    0.191 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.014    0.014    0.191    0.191 surrogate.py:1721(__call__)
        1    0.000    0.000    0.168    0.168 surrogate.py:934(__call__)
        1    0.007    0.007    0.140    0.140 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.055    0.055    0.055    0.055 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.037    0.037 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.037    0.037    0.037    0.037 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.035    0.017 surrogate.py:91(_splinterp_Cwrapper)
        2    0.034    0.017    0.034    0.017 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.027    0.005 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.027    0.005 surrogate.py:417(__call__)
       12    0.000    0.000    0.027    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.025    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.025    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.023    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.023    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.023    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.015    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.013    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.012    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.009    0.000 _base.py:297(predict)
       66    0.001    0.000    0.009    0.000 _base.py:287(_decision_function)
        5    0.009    0.002    0.009    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.005    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.004    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        4    0.002    0.000    0.002    0.001 _function_base_impl.py:1402(diff)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
       16    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
      132    0.000    0.000    0.002    0.000 _array_api.py:857(_asarray_with_order)
     2576    0.001    0.000    0.002    0.000 {built-in method builtins.isinstance}
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      198    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         28669 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 surrogate.py:934(__call__)
        6    0.000    0.000    0.026    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.026    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.026    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.025    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.025    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.023    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.023    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.022    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.014    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.013    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.012    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.009    0.000 _base.py:297(predict)
       66    0.001    0.000    0.009    0.000 _base.py:287(_decision_function)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.005    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.004    0.000 kernels.py:931(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.002    0.000 {built-in method builtins.isinstance}
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      198    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      198    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
     3894    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
      396    0.000    0.000    0.001    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         28669 function calls in 0.060 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.060    0.060 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.002    0.002    0.060    0.060 surrogate.py:1721(__call__)
        1    0.000    0.000    0.057    0.057 surrogate.py:934(__call__)
        1    0.001    0.001    0.030    0.030 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.027    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.027    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.026    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.025    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.025    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.023    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.023    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.023    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.014    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.013    0.000 _gpr.py:373(predict)
      132    0.002    0.000    0.012    0.000 validation.py:725(check_array)
        1    0.011    0.011    0.011    0.011 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.009    0.000 _base.py:297(predict)
       66    0.001    0.000    0.009    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.007    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.007    0.004    0.007    0.004 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.005    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.004    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.002    0.000 {built-in method builtins.isinstance}
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      198    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         94030 function calls in 0.151 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.152    0.152 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.001    0.001    0.152    0.152 surrogate.py:1721(__call__)
        1    0.002    0.002    0.147    0.147 surrogate.py:934(__call__)
       11    0.000    0.000    0.116    0.011 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.116    0.011 surrogate.py:417(__call__)
       19    0.000    0.000    0.116    0.006 surrogate.py:292(__call__)
      218    0.000    0.000    0.105    0.000 nodeFunction.py:205(__call__)
      218    0.003    0.000    0.105    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.096    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.096    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.096    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.003    0.000    0.060    0.000 validation.py:2793(validate_data)
      218    0.003    0.000    0.056    0.000 _gpr.py:373(predict)
      436    0.006    0.000    0.049    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.037    0.000 _base.py:297(predict)
      218    0.002    0.000    0.037    0.000 _base.py:287(_decision_function)
        1    0.002    0.002    0.027    0.027 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.024    0.000 kernels.py:833(__call__)
      436    0.005    0.000    0.019    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.018    0.000 kernels.py:931(__call__)
     1308    0.005    0.000    0.014    0.000 validation.py:371(_num_samples)
      218    0.005    0.000    0.011    0.000 kernels.py:1525(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
     8504    0.003    0.000    0.007    0.000 {built-in method builtins.isinstance}
      436    0.001    0.000    0.006    0.000 _array_api.py:857(_asarray_with_order)
      654    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      218    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.003 spline_interp_Cwrapper.py:50(interpolate)
      654    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      218    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      654    0.001    0.000    0.005    0.000 base.py:1166(__sklearn_tags__)
      436    0.001    0.000    0.005    0.000 fromnumeric.py:2304(sum)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      654    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
    12862    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         94026 function calls in 0.127 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.128    0.128 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.128    0.128 surrogate.py:1721(__call__)
        1    0.002    0.002    0.127    0.127 surrogate.py:934(__call__)
       11    0.000    0.000    0.117    0.011 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.117    0.011 surrogate.py:417(__call__)
       19    0.000    0.000    0.117    0.006 surrogate.py:292(__call__)
      218    0.000    0.000    0.106    0.000 nodeFunction.py:205(__call__)
      218    0.003    0.000    0.105    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.097    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.097    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.096    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.003    0.000    0.060    0.000 validation.py:2793(validate_data)
      218    0.003    0.000    0.056    0.000 _gpr.py:373(predict)
      436    0.006    0.000    0.049    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.038    0.000 _base.py:297(predict)
      218    0.002    0.000    0.038    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.024    0.000 kernels.py:833(__call__)
      436    0.005    0.000    0.019    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.018    0.000 kernels.py:931(__call__)
     1308    0.005    0.000    0.014    0.000 validation.py:371(_num_samples)
      218    0.005    0.000    0.011    0.000 kernels.py:1525(__call__)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
     8504    0.003    0.000    0.007    0.000 {built-in method builtins.isinstance}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      436    0.001    0.000    0.006    0.000 _array_api.py:857(_asarray_with_order)
      654    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      218    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      654    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      218    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.005    0.000 base.py:1166(__sklearn_tags__)
      436    0.001    0.000    0.005    0.000 fromnumeric.py:2304(sum)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      654    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
    12862    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
     1308    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
      436    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
      218    0.001    0.000    0.004    0.000 _py_warnings.py:254(filterwarnings)
      436    0.001    0.000    0.004    0.000 _base.py:711(__sklearn_tags__)
     1744    0.001    0.000    0.004    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         94030 function calls in 0.177 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.178    0.178 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.002    0.002    0.178    0.178 surrogate.py:1721(__call__)
        1    0.002    0.002    0.169    0.169 surrogate.py:934(__call__)
       11    0.000    0.000    0.115    0.010 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.115    0.010 surrogate.py:417(__call__)
       19    0.000    0.000    0.115    0.006 surrogate.py:292(__call__)
      218    0.000    0.000    0.105    0.000 nodeFunction.py:205(__call__)
      218    0.003    0.000    0.104    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.096    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.096    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.095    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.003    0.000    0.059    0.000 validation.py:2793(validate_data)
      218    0.003    0.000    0.056    0.000 _gpr.py:373(predict)
        1    0.002    0.002    0.051    0.051 surrogate.py:742(_coorbital_to_inertial_frame)
      436    0.006    0.000    0.049    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.037    0.000 _base.py:297(predict)
      218    0.002    0.000    0.037    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.024    0.000 kernels.py:833(__call__)
        1    0.000    0.000    0.021    0.021 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.020    0.020    0.021    0.021 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      436    0.005    0.000    0.019    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.018    0.000 kernels.py:931(__call__)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
     1308    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
      218    0.006    0.000    0.011    0.000 kernels.py:1525(__call__)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.009    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.009    0.005 spline_interp_Cwrapper.py:50(interpolate)
     8504    0.003    0.000    0.007    0.000 {built-in method builtins.isinstance}
      436    0.001    0.000    0.006    0.000 _array_api.py:857(_asarray_with_order)
      654    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      218    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      654    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.005    0.005 {method 'update' of 'dict' objects}
       18    0.005    0.000    0.005    0.000 surrogate.py:2126(<genexpr>)
      218    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      218    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      436    0.001    0.000    0.005    0.000 fromnumeric.py:2304(sum)
      654    0.001    0.000    0.005    0.000 base.py:1166(__sklearn_tags__)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         94026 function calls in 0.134 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.135    0.135 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.135    0.135 surrogate.py:1721(__call__)
        1    0.003    0.003    0.134    0.134 surrogate.py:934(__call__)
       11    0.000    0.000    0.121    0.011 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.121    0.011 surrogate.py:417(__call__)
       19    0.000    0.000    0.121    0.006 surrogate.py:292(__call__)
      218    0.000    0.000    0.108    0.000 nodeFunction.py:205(__call__)
      218    0.003    0.000    0.107    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.099    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.099    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.098    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.003    0.000    0.060    0.000 validation.py:2793(validate_data)
      218    0.003    0.000    0.058    0.000 _gpr.py:373(predict)
      436    0.006    0.000    0.049    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.038    0.000 _base.py:297(predict)
      218    0.002    0.000    0.037    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.025    0.000 kernels.py:833(__call__)
      436    0.005    0.000    0.019    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.019    0.000 kernels.py:931(__call__)
     1308    0.005    0.000    0.014    0.000 validation.py:371(_num_samples)
       23    0.013    0.001    0.013    0.001 {method 'dot' of 'numpy.ndarray' objects}
      218    0.007    0.000    0.012    0.000 kernels.py:1525(__call__)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
     8504    0.003    0.000    0.007    0.000 {built-in method builtins.isinstance}
      436    0.001    0.000    0.006    0.000 _array_api.py:857(_asarray_with_order)
      654    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      218    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      654    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      218    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.005    0.000 base.py:1166(__sklearn_tags__)
      436    0.001    0.000    0.005    0.000 fromnumeric.py:2304(sum)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      654    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
    12862    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
     1308    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
      436    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
      218    0.001    0.000    0.004    0.000 _py_warnings.py:254(filterwarnings)
      436    0.001    0.000    0.004    0.000 fromnumeric.py:66(_wrapreduction)
      436    0.001    0.000    0.004    0.000 _base.py:711(__sklearn_tags__)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         94006 function calls in 0.130 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.131    0.131 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.131    0.131 surrogate.py:1721(__call__)
        1    0.002    0.002    0.131    0.131 surrogate.py:934(__call__)
       11    0.000    0.000    0.118    0.011 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.118    0.011 surrogate.py:417(__call__)
       19    0.000    0.000    0.118    0.006 surrogate.py:292(__call__)
      218    0.000    0.000    0.106    0.000 nodeFunction.py:205(__call__)
      218    0.003    0.000    0.106    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.097    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.097    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.097    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.003    0.000    0.060    0.000 validation.py:2793(validate_data)
      218    0.003    0.000    0.057    0.000 _gpr.py:373(predict)
      436    0.006    0.000    0.049    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.037    0.000 _base.py:297(predict)
      218    0.002    0.000    0.037    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.024    0.000 kernels.py:833(__call__)
      436    0.005    0.000    0.019    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.019    0.000 kernels.py:931(__call__)
     1308    0.005    0.000    0.014    0.000 validation.py:371(_num_samples)
      218    0.006    0.000    0.012    0.000 kernels.py:1525(__call__)
       23    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
     8504    0.003    0.000    0.007    0.000 {built-in method builtins.isinstance}
      436    0.001    0.000    0.006    0.000 _array_api.py:857(_asarray_with_order)
      654    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      218    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      654    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      218    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.005    0.000 base.py:1166(__sklearn_tags__)
      436    0.001    0.000    0.005    0.000 fromnumeric.py:2304(sum)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      654    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
    12862    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
     1308    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
      436    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
      218    0.001    0.000    0.004    0.000 _py_warnings.py:254(filterwarnings)
      436    0.001    0.000    0.004    0.000 fromnumeric.py:66(_wrapreduction)
     1744    0.001    0.000    0.004    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         94006 function calls in 0.385 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.386    0.386 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.025    0.025    0.386    0.386 surrogate.py:1721(__call__)
        1    0.003    0.003    0.349    0.349 surrogate.py:934(__call__)
        1    0.009    0.009    0.232    0.232 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.114    0.010 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.114    0.010 surrogate.py:417(__call__)
       19    0.000    0.000    0.114    0.006 surrogate.py:292(__call__)
      218    0.000    0.000    0.103    0.000 nodeFunction.py:205(__call__)
      218    0.003    0.000    0.103    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.095    0.000 nodeFunction.py:96(__call__)
        1    0.000    0.000    0.095    0.095 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.094    0.094    0.095    0.095 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.095    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.094    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.069    0.069    0.069    0.069 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      436    0.003    0.000    0.058    0.000 validation.py:2793(validate_data)
      218    0.003    0.000    0.055    0.000 _gpr.py:373(predict)
        2    0.000    0.000    0.051    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.051    0.025    0.051    0.025 spline_interp_Cwrapper.py:50(interpolate)
      436    0.006    0.000    0.048    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.037    0.000 _base.py:297(predict)
      218    0.002    0.000    0.036    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.023    0.000 kernels.py:833(__call__)
      436    0.005    0.000    0.019    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.018    0.000 kernels.py:931(__call__)
     1308    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
        7    0.012    0.002    0.012    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.005    0.000    0.011    0.000 kernels.py:1525(__call__)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      436    0.001    0.000    0.006    0.000 _array_api.py:857(_asarray_with_order)
      654    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      218    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      218    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      654    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.005    0.000 base.py:1166(__sklearn_tags__)
      436    0.001    0.000    0.005    0.000 fromnumeric.py:2304(sum)
        4    0.004    0.001    0.005    0.001 _function_base_impl.py:1402(diff)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         94006 function calls in 0.123 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.124    0.124 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.124    0.124 surrogate.py:1721(__call__)
        1    0.002    0.002    0.124    0.124 surrogate.py:934(__call__)
       11    0.000    0.000    0.115    0.010 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.115    0.010 surrogate.py:417(__call__)
       19    0.000    0.000    0.115    0.006 surrogate.py:292(__call__)
      218    0.000    0.000    0.105    0.000 nodeFunction.py:205(__call__)
      218    0.003    0.000    0.105    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.096    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.096    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.096    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.003    0.000    0.060    0.000 validation.py:2793(validate_data)
      218    0.003    0.000    0.056    0.000 _gpr.py:373(predict)
      436    0.006    0.000    0.049    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.037    0.000 _base.py:297(predict)
      218    0.002    0.000    0.037    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.023    0.000 kernels.py:833(__call__)
      436    0.005    0.000    0.019    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.017    0.000 kernels.py:931(__call__)
     1308    0.005    0.000    0.014    0.000 validation.py:371(_num_samples)
      218    0.005    0.000    0.010    0.000 kernels.py:1525(__call__)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
     8504    0.003    0.000    0.007    0.000 {built-in method builtins.isinstance}
      436    0.001    0.000    0.006    0.000 _array_api.py:857(_asarray_with_order)
      654    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      218    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      654    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      218    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.005    0.000 base.py:1166(__sklearn_tags__)
      436    0.001    0.000    0.005    0.000 fromnumeric.py:2304(sum)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      654    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
    12862    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
     1308    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
      218    0.001    0.000    0.004    0.000 _py_warnings.py:254(filterwarnings)
      436    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
      436    0.001    0.000    0.004    0.000 _base.py:711(__sklearn_tags__)
     1744    0.001    0.000    0.004    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         94006 function calls in 0.187 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.188    0.188 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.002    0.002    0.188    0.188 surrogate.py:1721(__call__)
        1    0.002    0.002    0.184    0.184 surrogate.py:934(__call__)
       11    0.000    0.000    0.120    0.011 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.120    0.011 surrogate.py:417(__call__)
       19    0.000    0.000    0.120    0.006 surrogate.py:292(__call__)
      218    0.000    0.000    0.108    0.000 nodeFunction.py:205(__call__)
      218    0.003    0.000    0.107    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.099    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.099    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.098    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.003    0.000    0.061    0.000 validation.py:2793(validate_data)
        1    0.002    0.002    0.060    0.060 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.003    0.000    0.059    0.000 _gpr.py:373(predict)
      436    0.006    0.000    0.050    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.037    0.000 _base.py:297(predict)
      218    0.002    0.000    0.037    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.025    0.000 kernels.py:833(__call__)
        1    0.000    0.000    0.025    0.025 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.024    0.024    0.025    0.025 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      436    0.005    0.000    0.020    0.000 validation.py:103(_assert_all_finite)
        1    0.020    0.020    0.020    0.020 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.001    0.000    0.019    0.000 kernels.py:931(__call__)
     1308    0.005    0.000    0.014    0.000 validation.py:371(_num_samples)
      218    0.006    0.000    0.012    0.000 kernels.py:1525(__call__)
       23    0.012    0.001    0.012    0.001 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
     8504    0.003    0.000    0.007    0.000 {built-in method builtins.isinstance}
      436    0.001    0.000    0.006    0.000 _array_api.py:857(_asarray_with_order)
      218    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      218    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      654    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.005    0.000 base.py:1166(__sklearn_tags__)
      436    0.001    0.000    0.005    0.000 fromnumeric.py:2304(sum)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      654    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
    12862    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
```
