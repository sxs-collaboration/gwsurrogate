# GWSurrogate Evaluation Timing

Generated: 2026-07-23T14:10:55.379880+00:00

Times below are seconds per model evaluation. Raw repeats and context are in the JSON output.

PNG timing table: `test/benchmark_surrogate_evaluations.png`

## Summary

### master

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0166886` s, median `0.0171255` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0163653` s, median `0.0164523` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0173929` s, median `0.0177408` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0171366` s, median `0.0171902` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0145236` s, median `0.0147552` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0137508` s, median `0.0140062` s
- `dt=0.5 M`, `f_low=0`: best `0.0102394` s, median `0.0104146` s
- `dt=0.5 M`, `f_low=0.01`: best `0.01295` s, median `0.0130969` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0184923` s, median `0.0186258` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0180592` s, median `0.018195` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0198246` s, median `0.0199371` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0186912` s, median `0.0187276` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0227414` s, median `0.0229826` s
- `dt=0.1 M`, `f_low=0.01`: best `0.019345` s, median `0.0198975` s
- `dt=0.5 M`, `f_low=0`: best `0.0154385` s, median `0.0155816` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0178057` s, median `0.0180116` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0578879` s, median `0.0586496` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0245967` s, median `0.0248552` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0978081` s, median `0.0981032` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0257284` s, median `0.0258771` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0261575` s, median `0.0262269` s
- `dt=0.1 M`, `f_low=0.002`: best `0.296574` s, median `0.29696` s
- `dt=0.5 M`, `f_low=0.01`: best `0.023377` s, median `0.0235042` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0788431` s, median `0.0792088` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0214706` s, median `0.0215068` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.00709617` s, median `0.00721295` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0356385` s, median `0.035826` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.00820965` s, median `0.00834175` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.00843653` s, median `0.00859979` s
- `dt=0.1 M`, `f_low=0.002`: best `0.171151` s, median `0.171218` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0066688` s, median `0.00686194` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0380571` s, median `0.0381418` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.050525` s, median `0.0509061` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0280225` s, median `0.0283633` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0766255` s, median `0.0770339` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0294445` s, median `0.0298952` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0300659` s, median `0.030239` s
- `dt=0.1 M`, `f_low=0.002`: best `0.278025` s, median `0.279052` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0268972` s, median `0.0274162` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0799774` s, median `0.0801511` s

### PR-77

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0154001` s, median `0.0160313` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0155175` s, median `0.015732` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0166815` s, median `0.0169993` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0158799` s, median `0.0161857` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0135321` s, median `0.0138336` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0129638` s, median `0.013034` s
- `dt=0.5 M`, `f_low=0`: best `0.00936964` s, median `0.00946199` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0119162` s, median `0.0120891` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0182585` s, median `0.0183824` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0175673` s, median `0.0177198` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0196836` s, median `0.0198147` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0185785` s, median `0.0187413` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.02238` s, median `0.0225103` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0191737` s, median `0.0192679` s
- `dt=0.5 M`, `f_low=0`: best `0.015232` s, median `0.0154676` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0175614` s, median `0.0178109` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0550963` s, median `0.055369` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0243786` s, median `0.0245393` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0940871` s, median `0.0955654` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0256134` s, median `0.0261751` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0261347` s, median `0.0262079` s
- `dt=0.1 M`, `f_low=0.002`: best `0.294931` s, median `0.29899` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0237944` s, median `0.0238841` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0798591` s, median `0.0799572` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0216708` s, median `0.0217908` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0073287` s, median `0.00757607` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0359413` s, median `0.0359842` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.00836174` s, median `0.00879171` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.00915493` s, median `0.0092624` s
- `dt=0.1 M`, `f_low=0.002`: best `0.169999` s, median `0.170747` s
- `dt=0.5 M`, `f_low=0.01`: best `0.00642658` s, median `0.00692738` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0380474` s, median `0.0381064` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0517805` s, median `0.0519807` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.029187` s, median `0.0295873` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0766836` s, median `0.0776007` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0309215` s, median `0.0315047` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0314038` s, median `0.0316432` s
- `dt=0.1 M`, `f_low=0.002`: best `0.279332` s, median `0.279551` s
- `dt=0.5 M`, `f_low=0.01`: best `0.028008` s, median `0.0287961` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0805586` s, median `0.0808575` s

## Context

### master

- Git branch: `master`
- Git commit: `64f9f64870a90c2557dbbac1ec8dcde7ebc7e310`
- Git describe: `v1.1.8-43-g64f9f64`
- Python: `3.14.6 (main, Jun 10 2026, 14:29:35) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1020-azure x86_64`
- CPU count: `4`
- Conda env: `unknown`

Submodules:

- `gwsurrogate/eval_pysur`: `6a51ecba0ed8ddc26a85e5d2918596aa9f58f534` initialized ((heads/master))

### PR-77

- Git branch: `unknown`
- Git commit: `526b682335d6d32015f7773d48dd915a76b9798c`
- Git describe: `v1.1.8-49-g526b682`
- Python: `3.14.6 (main, Jun 10 2026, 14:29:35) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1020-azure x86_64`
- CPU count: `4`
- Conda env: `unknown`

Submodules:

- `gwsurrogate/eval_pysur`: `6a51ecba0ed8ddc26a85e5d2918596aa9f58f534` initialized ((heads/master))

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
Model name:                              AMD EPYC 9V74 80-Core Processor
CPU family:                              25
Model:                                   17
Thread(s) per core:                      2
Core(s) per socket:                      2
Socket(s):                               1
Stepping:                                1
BogoMIPS:                                5192.26
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves user_shstk clzero xsaveerptr rdpru arat npt nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload umip vaes vpclmulqdq rdpid fsrm
Virtualization:                          AMD-V
Hypervisor vendor:                       Microsoft
Virtualization type:                     full
L1d cache:                               64 KiB (2 instances)
L1i cache:                               64 KiB (2 instances)
L2 cache:                                2 MiB (2 instances)
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

#### PR-77

lscpu:

```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           48 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  4
On-line CPU(s) list:                     0-3
Vendor ID:                               AuthenticAMD
Model name:                              AMD EPYC 9V74 80-Core Processor
CPU family:                              25
Model:                                   17
Thread(s) per core:                      2
Core(s) per socket:                      2
Socket(s):                               1
Stepping:                                1
BogoMIPS:                                5192.26
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves user_shstk clzero xsaveerptr rdpru arat npt nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload umip vaes vpclmulqdq rdpid fsrm
Virtualization:                          AMD-V
Hypervisor vendor:                       Microsoft
Virtualization type:                     full
L1d cache:                               64 KiB (2 instances)
L1i cache:                               64 KiB (2 instances)
L2 cache:                                2 MiB (2 instances)
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
         14341 function calls in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.022    0.022 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.022    0.022 surrogate.py:1721(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
       42    0.001    0.000    0.004    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5592(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1917(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      295    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:817(rotate_spin)
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         14342 function calls in 0.021 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.021    0.021 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.021    0.021 surrogate.py:1721(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
       42    0.001    0.000    0.004    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5592(append)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1917(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      295    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1006    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         14341 function calls in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.022    0.022 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.022    0.022 surrogate.py:1721(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
       42    0.001    0.000    0.004    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5592(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1917(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      295    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         14342 function calls in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.022    0.022 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.022    0.022 surrogate.py:1721(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5592(append)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1917(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      295    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         9295 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 85 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1721(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:922(__call__)
        4    0.000    0.000    0.006    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.006    0.006 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.006    0.006    0.006    0.006 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:754(__call__)
       42    0.001    0.000    0.004    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
      238    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5592(append)
      238    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1917(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      248    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      534    0.000    0.000    0.000    0.000 {built-in method numpy.array}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:817(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:827(coorb_spins_from_copr_spins)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      504    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:849(normalize_spin)
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         14878 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 97 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1721(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:754(__call__)
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
       42    0.001    0.000    0.004    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5592(append)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1917(ravel)
      365    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      862    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:817(rotate_spin)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         9295 function calls in 0.014 seconds

   Ordered by: cumulative time
   List reduced from 85 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.014    0.014 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.014    0.014 surrogate.py:1721(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:754(__call__)
       42    0.001    0.000    0.004    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
      238    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:832(inertial_waveform_modes)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      238    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1917(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      248    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      534    0.000    0.000    0.000    0.000 {built-in method numpy.array}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:817(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:827(coorb_spins_from_copr_spins)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      504    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         14878 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 97 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1721(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:754(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
       42    0.001    0.000    0.004    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5592(append)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1917(ravel)
      365    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      862    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1006    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         13382 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:1252(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:607(_integrate_forward)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      221    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      963    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      191    0.000    0.000    0.000    0.000 fromnumeric.py:1917(ravel)
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         13383 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:1252(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:607(_integrate_forward)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
      963    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      221    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      191    0.000    0.000    0.000    0.000 fromnumeric.py:1917(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         13382 function calls in 0.025 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:1252(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:607(_integrate_forward)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
      221    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      963    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      191    0.000    0.000    0.000    0.000 fromnumeric.py:1917(ravel)
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         13383 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:1252(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:373(__call__)
      546    0.001    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:607(_integrate_forward)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:508(_initialize)
      963    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      221    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      191    0.000    0.000    0.000    0.000 fromnumeric.py:1917(ravel)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         7669 function calls in 0.025 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.025    0.025 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.025    0.025 surrogate.py:1721(__call__)
        1    0.000    0.000    0.025    0.025 precessing_surrogate.py:1252(__call__)
        1    0.000    0.000    0.025    0.025 precessing_surrogate.py:922(__call__)
        4    0.000    0.000    0.011    0.003 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:607(_integrate_forward)
      505    0.000    0.000    0.006    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      505    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5592(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      623    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1917(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:817(rotate_spin)
       92    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:827(coorb_spins_from_copr_spins)
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:849(normalize_spin)
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        6    0.000    0.000    0.000    0.000 fromnumeric.py:2342(sum)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         14460 function calls in 0.025 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.025    0.025 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.025    0.025 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:1252(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:607(_integrate_forward)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:641(_integrate_backward)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:326(_get_t_from_omega)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5592(append)
      360    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
      360    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     1102    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:508(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      191    0.000    0.000    0.000    0.000 fromnumeric.py:1917(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         7669 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1721(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:1252(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:607(_integrate_forward)
      505    0.000    0.000    0.006    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      505    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5592(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      623    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1917(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:817(rotate_spin)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       92    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:827(coorb_spins_from_copr_spins)
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:849(normalize_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         14460 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:1252(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:922(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:607(_integrate_forward)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:832(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:641(_integrate_backward)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:326(_get_t_from_omega)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5592(append)
      360    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:839(splinterp_many)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      360    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     1102    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:508(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      191    0.000    0.000    0.000    0.000 fromnumeric.py:1917(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         9044 function calls in 0.061 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.061    0.061 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.061    0.061 surrogate.py:1721(__call__)
        1    0.003    0.003    0.055    0.055 surrogate.py:934(__call__)
        1    0.001    0.001    0.031    0.031 surrogate.py:742(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.020    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.020    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.014    0.014 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.013    0.013    0.014    0.014 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.005    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 {method 'update' of 'dict' objects}
       21    0.003    0.000    0.003    0.000 surrogate.py:2126(<genexpr>)
      177    0.001    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        3    0.001    0.000    0.001    0.000 {built-in method numpy.ascontiguousarray}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         9040 function calls in 0.029 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.029    0.029 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.002    0.002    0.028    0.028 surrogate.py:934(__call__)
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         9044 function calls in 0.099 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.099    0.099 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.099    0.099 surrogate.py:1721(__call__)
        1    0.003    0.003    0.079    0.079 surrogate.py:934(__call__)
        1    0.003    0.003    0.056    0.056 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.022    0.022 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.021    0.021    0.022    0.022 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.020    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.020    0.001 surrogate.py:292(__call__)
        1    0.018    0.018    0.018    0.018 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.015    0.015 {method 'update' of 'dict' objects}
       21    0.015    0.001    0.015    0.001 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        4    0.002    0.001    0.002    0.001 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         9040 function calls in 0.030 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.030    0.030 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.030    0.030 surrogate.py:1721(__call__)
        1    0.002    0.002    0.029    0.029 surrogate.py:934(__call__)
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.006    0.000    0.006    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         9017 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.002    0.002    0.030    0.030 surrogate.py:934(__call__)
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.006    0.000    0.006    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         9017 function calls in 0.298 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.297    0.297 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.024    0.024    0.297    0.297 surrogate.py:1721(__call__)
        1    0.003    0.003    0.263    0.263 surrogate.py:934(__call__)
        1    0.010    0.010    0.240    0.240 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.100    0.100 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.100    0.100    0.100    0.100 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.074    0.074    0.074    0.074 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.048    0.024 surrogate.py:91(_splinterp_Cwrapper)
        2    0.047    0.024    0.048    0.024 spline_interp_Cwrapper.py:50(interpolate)
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.011    0.000 nodeFunction.py:125(__call__)
        9    0.011    0.001    0.011    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         9017 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.002    0.002    0.027    0.027 surrogate.py:934(__call__)
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         9017 function calls in 0.084 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.084    0.084 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.084    0.084 surrogate.py:1721(__call__)
        1    0.002    0.002    0.079    0.079 surrogate.py:934(__call__)
        1    0.002    0.002    0.057    0.057 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.025    0.025 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.024    0.024    0.025    0.025 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.020    0.020    0.020    0.020 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
       26    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         3540 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.021    0.021 surrogate.py:934(__call__)
        1    0.001    0.001    0.016    0.016 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.005    0.005    0.005    0.005 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        1    0.000    0.000    0.005    0.005 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.004    0.004    0.005    0.005 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
        2    0.000    0.000    0.004    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.004    0.002 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
       11    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         3536 function calls in 0.009 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.009    0.009 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.009    0.009 surrogate.py:1721(__call__)
        1    0.000    0.000    0.008    0.008 surrogate.py:934(__call__)
        6    0.000    0.000    0.004    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.004    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         3540 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.037    0.037 surrogate.py:1721(__call__)
        1    0.000    0.000    0.033    0.033 surrogate.py:934(__call__)
        1    0.001    0.001    0.028    0.028 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.010    0.010    0.010    0.010 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.007    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.007    0.003    0.007    0.004 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         3536 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1721(__call__)
        1    0.000    0.000    0.009    0.009 surrogate.py:934(__call__)
        1    0.000    0.000    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.004    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.004    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         3523 function calls in 0.011 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.011    0.011 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.011    0.011 surrogate.py:1721(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:934(__call__)
        1    0.000    0.000    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.004    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.004    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         3523 function calls in 0.174 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.174    0.174 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.011    0.011    0.174    0.174 surrogate.py:1721(__call__)
        1    0.000    0.000    0.156    0.156 surrogate.py:934(__call__)
        1    0.007    0.007    0.151    0.151 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.061    0.061    0.061    0.061 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.041    0.041 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.041    0.041    0.041    0.041 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.037    0.018 surrogate.py:91(_splinterp_Cwrapper)
        2    0.036    0.018    0.037    0.018 spline_interp_Cwrapper.py:50(interpolate)
        5    0.007    0.001    0.007    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1413(diff)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       20    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         3523 function calls in 0.008 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.008    0.008 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.008    0.008 surrogate.py:1721(__call__)
        1    0.000    0.000    0.008    0.008 surrogate.py:934(__call__)
        6    0.000    0.000    0.004    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.004    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
        1    0.000    0.000    0.003    0.003 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         3523 function calls in 0.040 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.040    0.040 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.040    0.040 surrogate.py:1721(__call__)
        1    0.000    0.000    0.038    0.038 surrogate.py:934(__call__)
        1    0.001    0.001    0.033    0.033 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.012    0.012    0.012    0.012 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.009    0.009 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.009    0.009 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.008    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         10972 function calls in 0.057 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.057    0.057 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.057    0.057 surrogate.py:1721(__call__)
        1    0.002    0.002    0.053    0.053 surrogate.py:934(__call__)
        1    0.001    0.001    0.025    0.025 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.005    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       18    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         10968 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1721(__call__)
        1    0.002    0.002    0.033    0.033 surrogate.py:934(__call__)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         10972 function calls in 0.082 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.082    0.082 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.082    0.082 surrogate.py:1721(__call__)
        1    0.002    0.002    0.074    0.074 surrogate.py:934(__call__)
        1    0.002    0.002    0.048    0.048 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.019    0.019 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.019    0.019    0.019    0.019 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.016    0.016    0.016    0.016 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.008    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       18    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         10968 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1721(__call__)
        1    0.002    0.002    0.034    0.034 surrogate.py:934(__call__)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         10948 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1721(__call__)
        1    0.002    0.002    0.034    0.034 surrogate.py:934(__call__)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         10948 function calls in 0.281 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.281    0.281 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.020    0.020    0.281    0.281 surrogate.py:1721(__call__)
        1    0.003    0.003    0.251    0.251 surrogate.py:934(__call__)
        1    0.008    0.008    0.223    0.223 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.092    0.092 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.092    0.092    0.092    0.092 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.070    0.070    0.070    0.070 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.045    0.022 surrogate.py:91(_splinterp_Cwrapper)
        2    0.044    0.022    0.045    0.022 spline_interp_Cwrapper.py:50(interpolate)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
        7    0.011    0.002    0.011    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1413(diff)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5592(append)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         10948 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1721(__call__)
        1    0.002    0.002    0.032    0.032 surrogate.py:934(__call__)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         10948 function calls in 0.084 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.084    0.084 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.084    0.084 surrogate.py:1721(__call__)
        1    0.002    0.002    0.081    0.081 surrogate.py:934(__call__)
        1    0.002    0.002    0.054    0.054 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.022    0.022 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.022    0.022 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.019    0.019    0.019    0.019 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

#### PR-77

##### NRSur7dq4 / mks_dt_0.000244140625_flow_0

```text
         9221 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1721(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:754(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      750    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         9222 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1721(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:754(__call__)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      750    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         9221 function calls in 0.020 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.020    0.020 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.020    0.020 surrogate.py:1721(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      750    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         9222 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1721(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:754(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      750    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
       22    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         4175 function calls in 0.016 seconds

   Ordered by: cumulative time
   List reduced from 85 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.016    0.016 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.016    0.016 surrogate.py:1721(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:940(__call__)
        4    0.000    0.000    0.006    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.006    0.006 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.006    0.006    0.006    0.006 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:607(_integrate_forward)
      238    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
      238    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      486    0.001    0.000    0.001    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      492    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:868(normalize_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:844(coorb_spins_from_copr_spins)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
      111    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:824(rotate_spin)
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         9758 function calls in 0.017 seconds

   Ordered by: cumulative time
   List reduced from 97 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.017    0.017 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.017    0.017 surrogate.py:1721(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      820    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
      254    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         4175 function calls in 0.011 seconds

   Ordered by: cumulative time
   List reduced from 85 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.011    0.011 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.011    0.011 surrogate.py:1721(__call__)
        1    0.000    0.000    0.011    0.011 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:607(_integrate_forward)
      238    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:851(inertial_waveform_modes)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
      238    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      486    0.001    0.000    0.001    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      492    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:844(coorb_spins_from_copr_spins)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:868(normalize_spin)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:824(rotate_spin)
       13    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:112(<genexpr>)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         9758 function calls in 0.016 seconds

   Ordered by: cumulative time
   List reduced from 97 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.016    0.016 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.016    0.016 surrogate.py:1721(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      820    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
      650    0.000    0.000    0.000    0.000 __init__.py:271(POINTER)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         11494 function calls in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.022    0.022 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.022    0.022 surrogate.py:1721(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:373(__call__)
      546    0.001    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:607(_integrate_forward)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      943    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         11495 function calls in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.022    0.022 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.022    0.022 surrogate.py:1721(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:607(_integrate_forward)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
      943    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         11494 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:607(_integrate_forward)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      943    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         11495 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:373(__call__)
      546    0.001    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:607(_integrate_forward)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      943    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         5781 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:940(__call__)
        4    0.000    0.000    0.011    0.003 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:607(_integrate_forward)
      505    0.000    0.000    0.006    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      505    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      603    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:868(normalize_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:844(coorb_spins_from_copr_spins)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:824(rotate_spin)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         12572 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:607(_integrate_forward)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:641(_integrate_backward)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:326(_get_t_from_omega)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      360    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     1082    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:508(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         5781 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1721(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:607(_integrate_forward)
      505    0.000    0.000    0.006    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      505    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      603    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:868(normalize_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:844(coorb_spins_from_copr_spins)
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:824(rotate_spin)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         12572 function calls in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.022    0.022 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.022    0.022 surrogate.py:1721(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:607(_integrate_forward)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:641(_integrate_backward)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
     1082    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      360    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         9044 function calls in 0.058 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.058    0.058 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.058    0.058 surrogate.py:1721(__call__)
        1    0.003    0.003    0.054    0.054 surrogate.py:934(__call__)
        1    0.001    0.001    0.031    0.031 surrogate.py:742(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.020    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.020    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.014    0.014 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.013    0.013    0.014    0.014 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.005    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       21    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
        9    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        3    0.001    0.000    0.001    0.000 {built-in method numpy.ascontiguousarray}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         9040 function calls in 0.029 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.029    0.029 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.002    0.002    0.028    0.028 surrogate.py:934(__call__)
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.018    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
       26    0.006    0.000    0.006    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         9044 function calls in 0.097 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.097    0.097 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.097    0.097 surrogate.py:1721(__call__)
        1    0.003    0.003    0.080    0.080 surrogate.py:934(__call__)
        1    0.003    0.003    0.056    0.056 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.020    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.020    0.001 surrogate.py:292(__call__)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.013    0.013 {method 'update' of 'dict' objects}
       21    0.013    0.001    0.013    0.001 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.011    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.009    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.009    0.005 spline_interp_Cwrapper.py:50(interpolate)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.003    0.001    0.003    0.001 _function_base_impl.py:1413(diff)
      177    0.001    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         9040 function calls in 0.030 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.030    0.030 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.030    0.030 surrogate.py:1721(__call__)
        1    0.002    0.002    0.029    0.029 surrogate.py:934(__call__)
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.006    0.000    0.006    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         9017 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.002    0.002    0.031    0.031 surrogate.py:934(__call__)
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      531    0.000    0.000    0.000    0.000 {built-in method _warnings._acquire_lock}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         9017 function calls in 0.299 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.299    0.299 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.024    0.024    0.299    0.299 surrogate.py:1721(__call__)
        1    0.003    0.003    0.264    0.264 surrogate.py:934(__call__)
        1    0.010    0.010    0.240    0.240 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.100    0.100 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.100    0.100    0.100    0.100 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.074    0.074    0.074    0.074 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.048    0.024 surrogate.py:91(_splinterp_Cwrapper)
        2    0.047    0.024    0.048    0.024 spline_interp_Cwrapper.py:50(interpolate)
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
        9    0.011    0.001    0.011    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.011    0.000 nodeFunction.py:125(__call__)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         9017 function calls in 0.029 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.029    0.029 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.002    0.002    0.028    0.028 surrogate.py:934(__call__)
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         9017 function calls in 0.083 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.083    0.083 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.083    0.083 surrogate.py:1721(__call__)
        1    0.002    0.002    0.079    0.079 surrogate.py:934(__call__)
        1    0.002    0.002    0.057    0.057 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.025    0.025 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.024    0.024    0.025    0.025 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.020    0.020    0.020    0.020 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
      177    0.001    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      531    0.000    0.000    0.000    0.000 {built-in method _warnings._filters_mutated_lock_held}
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         3540 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.021    0.021 surrogate.py:934(__call__)
        1    0.001    0.001    0.016    0.016 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.005    0.005    0.005    0.005 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
        1    0.000    0.000    0.005    0.005 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.004    0.004    0.005    0.005 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
        2    0.000    0.000    0.004    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.004    0.002 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
       11    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         3536 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1721(__call__)
        1    0.000    0.000    0.009    0.009 surrogate.py:934(__call__)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        1    0.000    0.000    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         3540 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.037    0.037 surrogate.py:1721(__call__)
        1    0.000    0.000    0.034    0.034 surrogate.py:934(__call__)
        1    0.001    0.001    0.029    0.029 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.010    0.010    0.010    0.010 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.007    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.007    0.003    0.007    0.004 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         3536 function calls in 0.011 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.011    0.011 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.011    0.011 surrogate.py:1721(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:934(__call__)
        1    0.000    0.000    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.004    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         3523 function calls in 0.011 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.011    0.011 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.011    0.011 surrogate.py:1721(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:934(__call__)
        1    0.000    0.000    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         3523 function calls in 0.171 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.171    0.171 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.011    0.011    0.171    0.171 surrogate.py:1721(__call__)
        1    0.000    0.000    0.154    0.154 surrogate.py:934(__call__)
        1    0.006    0.006    0.149    0.149 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.061    0.061    0.061    0.061 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.041    0.041 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.041    0.041    0.041    0.041 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.037    0.018 surrogate.py:91(_splinterp_Cwrapper)
        2    0.036    0.018    0.037    0.018 spline_interp_Cwrapper.py:50(interpolate)
        5    0.006    0.001    0.006    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1413(diff)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       20    0.000    0.000    0.000    0.000 __init__.py:613(cast)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         3523 function calls in 0.009 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.009    0.009 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.009    0.009 surrogate.py:1721(__call__)
        1    0.000    0.000    0.009    0.009 surrogate.py:934(__call__)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         3523 function calls in 0.040 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.040    0.040 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.040    0.040 surrogate.py:1721(__call__)
        1    0.000    0.000    0.038    0.038 surrogate.py:934(__call__)
        1    0.001    0.001    0.033    0.033 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.012    0.012    0.012    0.012 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.009    0.009 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.009    0.009 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.008    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
      332    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         10972 function calls in 0.060 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.060    0.060 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.060    0.060 surrogate.py:1721(__call__)
        1    0.002    0.002    0.055    0.055 surrogate.py:934(__call__)
        1    0.001    0.001    0.027    0.027 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.017    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.003 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       18    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         10968 function calls in 0.036 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.036    0.036 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.036    0.036 surrogate.py:1721(__call__)
        1    0.002    0.002    0.035    0.035 surrogate.py:934(__call__)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         10972 function calls in 0.083 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.083    0.083 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.083    0.083 surrogate.py:1721(__call__)
        1    0.002    0.002    0.076    0.076 surrogate.py:934(__call__)
        1    0.002    0.002    0.048    0.048 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.019    0.019 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.019    0.019    0.019    0.019 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
        1    0.016    0.016    0.016    0.016 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.009    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.008    0.004    0.009    0.004 spline_interp_Cwrapper.py:50(interpolate)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       18    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         10968 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.037    0.037 surrogate.py:1721(__call__)
        1    0.002    0.002    0.036    0.036 surrogate.py:934(__call__)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         10948 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.037    0.037 surrogate.py:1721(__call__)
        1    0.002    0.002    0.037    0.037 surrogate.py:934(__call__)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         10948 function calls in 0.281 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.281    0.281 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.020    0.020    0.281    0.281 surrogate.py:1721(__call__)
        1    0.003    0.003    0.251    0.251 surrogate.py:934(__call__)
        1    0.009    0.009    0.222    0.222 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.091    0.091 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.091    0.091    0.091    0.091 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.070    0.070    0.070    0.070 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.046    0.023 surrogate.py:91(_splinterp_Cwrapper)
        2    0.045    0.023    0.046    0.023 spline_interp_Cwrapper.py:50(interpolate)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        7    0.010    0.001    0.010    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.003    0.001    0.003    0.001 _function_base_impl.py:1413(diff)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        6    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         10948 function calls in 0.033 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.033    0.033 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.033    0.033 surrogate.py:1721(__call__)
        1    0.002    0.002    0.032    0.032 surrogate.py:934(__call__)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
      654    0.000    0.000    0.000    0.000 {built-in method _warnings._acquire_lock}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         10948 function calls in 0.085 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.085    0.085 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.085    0.085 surrogate.py:1721(__call__)
        1    0.002    0.002    0.082    0.082 surrogate.py:934(__call__)
        1    0.002    0.002    0.054    0.054 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.022    0.022 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.022    0.022 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.019    0.019    0.019    0.019 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```
