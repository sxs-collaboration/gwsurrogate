# GWSurrogate Evaluation Timing

Generated: 2026-05-07T22:32:35.312059+00:00

Times below are seconds per model evaluation. Raw repeats and context are in the JSON output.

PNG timing table: `test/benchmark_surrogate_evaluations.png`

## Summary

### master

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0230691` s, median `0.0231715` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0227581` s, median `0.0231429` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0256818` s, median `0.0263388` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0255409` s, median `0.0256009` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0208452` s, median `0.0210797` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0211951` s, median `0.021295` s
- `dt=0.5 M`, `f_low=0`: best `0.0167155` s, median `0.0168583` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0203866` s, median `0.020475` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0335445` s, median `0.0336837` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0322648` s, median `0.03273` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0346448` s, median `0.0347667` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0334675` s, median `0.0336115` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0358141` s, median `0.0358866` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0341992` s, median `0.0342661` s
- `dt=0.5 M`, `f_low=0`: best `0.029328` s, median `0.0295406` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0329334` s, median `0.0333334` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0621461` s, median `0.062327` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0286073` s, median `0.0289289` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.102858` s, median `0.104108` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.029929` s, median `0.0301968` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0305747` s, median `0.0309182` s
- `dt=0.1 M`, `f_low=0.002`: best `0.30831` s, median `0.31099` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0267431` s, median `0.0272417` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0857588` s, median `0.0863894` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0213289` s, median `0.0214674` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.00868207` s, median `0.00915559` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.035894` s, median `0.0361214` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.00906266` s, median `0.0091516` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.00949876` s, median `0.00957251` s
- `dt=0.1 M`, `f_low=0.002`: best `0.163521` s, median `0.165924` s
- `dt=0.5 M`, `f_low=0.01`: best `0.00728117` s, median `0.00733727` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0364216` s, median `0.0367602` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0583414` s, median `0.0591148` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0332942` s, median `0.0347119` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0872168` s, median `0.0894919` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0363495` s, median `0.0365358` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0366169` s, median `0.0370531` s
- `dt=0.1 M`, `f_low=0.002`: best `0.293235` s, median `0.293916` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0326481` s, median `0.0327732` s
- `dt=0.5 M`, `f_low=0.002`: best `0.087996` s, median `0.0883886` s

### PR-73

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0192651` s, median `0.0200937` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0193969` s, median `0.0194641` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0209878` s, median `0.0209995` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0202072` s, median `0.0202591` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0164512` s, median `0.0165797` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0169944` s, median `0.0170597` s
- `dt=0.5 M`, `f_low=0`: best `0.0126637` s, median `0.012724` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0160026` s, median `0.0163468` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0244377` s, median `0.0246728` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0237122` s, median `0.0239089` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0254963` s, median `0.0258634` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0243277` s, median `0.0244981` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0267965` s, median `0.0268536` s
- `dt=0.1 M`, `f_low=0.01`: best `0.025052` s, median `0.0253557` s
- `dt=0.5 M`, `f_low=0`: best `0.0202715` s, median `0.0204482` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0238992` s, median `0.0240477` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0602658` s, median `0.060415` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0279224` s, median `0.0281697` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.103315` s, median `0.103659` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0296869` s, median `0.0299907` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0299766` s, median `0.0300119` s
- `dt=0.1 M`, `f_low=0.002`: best `0.309807` s, median `0.311177` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0271899` s, median `0.0274515` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0858793` s, median `0.0863163` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.021575` s, median `0.0217269` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.00780499` s, median `0.00799616` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0354043` s, median `0.035666` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.00899995` s, median `0.00913684` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.00857997` s, median `0.00874788` s
- `dt=0.1 M`, `f_low=0.002`: best `0.162419` s, median `0.163346` s
- `dt=0.5 M`, `f_low=0.01`: best `0.00731093` s, median `0.00747605` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0373259` s, median `0.0375041` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0575803` s, median `0.0586347` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0327412` s, median `0.0334568` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.086917` s, median `0.0873492` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0355544` s, median `0.0358397` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0347476` s, median `0.0353128` s
- `dt=0.1 M`, `f_low=0.002`: best `0.290393` s, median `0.290944` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0333319` s, median `0.0335934` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0905061` s, median `0.0908575` s

## Context

### master

- Git branch: `master`
- Git commit: `1946a24f8541fe4471c2b48acc7894838b46332e`
- Git describe: `v1.1.8-27-g1946a24`
- Python: `3.14.4 (main, Apr  8 2026, 02:27:22) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1010-azure x86_64`
- CPU count: `4`
- Conda env: `unknown`

Submodules:

- `gwsurrogate/eval_pysur`: `6a51ecba0ed8ddc26a85e5d2918596aa9f58f534` initialized ((heads/master))

### PR-73

- Git branch: `unknown`
- Git commit: `aee4424d6d985d6e060e5c81008e8dd36dfbd9eb`
- Git describe: `v1.1.8-30-gaee4424`
- Python: `3.14.4 (main, Apr  8 2026, 02:27:22) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1010-azure x86_64`
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

#### PR-73

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
         27100 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1721(__call__)
        1    0.000    0.000    0.032    0.032 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:847(inertial_waveform_modes)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:110(rotateWaveform)
        1    0.006    0.006    0.007    0.007 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      574    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     1633    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
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
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:847(inertial_waveform_modes)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:110(rotateWaveform)
        1    0.006    0.006    0.007    0.007 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      574    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
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
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         27100 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1721(__call__)
        1    0.000    0.000    0.034    0.034 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:110(rotateWaveform)
        1    0.007    0.007    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
     1633    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         27101 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:346(evaluate_case)
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
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      574    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
     1633    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         20326 function calls in 0.027 seconds

   Ordered by: cumulative time
   List reduced from 91 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.027    0.027 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.027    0.027 surrogate.py:1721(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:626(_integrate_forward)
      238    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        4    0.000    0.000    0.006    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
        1    0.000    0.000    0.005    0.005 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.005    0.005    0.005    0.005 spline_interp_Cwrapper.py:123(interpolate_many_complex)
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
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         27707 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_forward)
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
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         20326 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 91 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:626(_integrate_forward)
      238    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:769(__call__)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2390    0.001    0.000    0.004    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
     2390    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
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
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         27707 function calls in 0.030 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.030    0.030 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.030    0.030 surrogate.py:1721(__call__)
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2876    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1703    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         43096 function calls in 0.047 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
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
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         43097 function calls in 0.046 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.046    0.046 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.046    0.046 surrogate.py:1721(__call__)
        1    0.000    0.000    0.045    0.045 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.045    0.045 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.012    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     2605    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
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
        1    0.000    0.000    0.048    0.048 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.048    0.048 surrogate.py:1721(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.012    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
       33    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         43097 function calls in 0.047 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.012    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
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
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         35075 function calls in 0.046 seconds

   Ordered by: cumulative time
   List reduced from 94 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.046    0.046 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.046    0.046 surrogate.py:1721(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.019    0.019 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.018    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.003    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.003    0.000    0.011    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
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
      597    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
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
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         44452 function calls in 0.048 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.048    0.048 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.048    0.048 surrogate.py:1721(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     5274    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
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
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         35075 function calls in 0.040 seconds

   Ordered by: cumulative time
   List reduced from 94 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.040    0.040 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.040    0.040 surrogate.py:1721(__call__)
        1    0.000    0.000    0.040    0.040 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.039    0.039 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.019    0.019 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.018    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.003    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.003    0.000    0.011    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
     4637    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.001    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     2142    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:561(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:864(normalize_spin)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         44452 function calls in 0.046 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.046    0.046 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.046    0.046 surrogate.py:1721(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     5274    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:660(_integrate_backward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:626(_integrate_forward)
     5274    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
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
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         9044 function calls in 0.066 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.066    0.066 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.066    0.066 surrogate.py:1721(__call__)
        1    0.004    0.004    0.061    0.061 surrogate.py:934(__call__)
        1    0.001    0.001    0.033    0.033 surrogate.py:742(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
        1    0.000    0.000    0.015    0.015 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.014    0.014    0.015    0.015 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       26    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.003    0.005    0.003 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       21    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1402(diff)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        9    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        3    0.001    0.000    0.001    0.000 {built-in method numpy.ascontiguousarray}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         9040 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1721(__call__)
        1    0.002    0.002    0.033    0.033 surrogate.py:934(__call__)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         9044 function calls in 0.106 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.106    0.106 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.106    0.106 surrogate.py:1721(__call__)
        1    0.004    0.004    0.086    0.086 surrogate.py:934(__call__)
        1    0.003    0.003    0.059    0.059 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.024    0.024 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.023    0.023    0.024    0.024 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.015    0.015 {method 'update' of 'dict' objects}
       21    0.015    0.001    0.015    0.001 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
       26    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        4    0.002    0.001    0.002    0.001 _function_base_impl.py:1402(diff)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         9040 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1721(__call__)
        1    0.002    0.002    0.034    0.034 surrogate.py:934(__call__)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         9017 function calls in 0.036 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.036    0.036 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.036    0.036 surrogate.py:1721(__call__)
        1    0.002    0.002    0.035    0.035 surrogate.py:934(__call__)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         9017 function calls in 0.311 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.311    0.311 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.027    0.027    0.311    0.311 surrogate.py:1721(__call__)
        1    0.004    0.004    0.271    0.271 surrogate.py:934(__call__)
        1    0.010    0.010    0.244    0.244 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.100    0.100 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.100    0.100    0.100    0.100 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.073    0.073    0.073    0.073 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.053    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.052    0.026    0.053    0.026 spline_interp_Cwrapper.py:50(interpolate)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
        9    0.013    0.001    0.013    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         9017 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1721(__call__)
        1    0.002    0.002    0.031    0.031 surrogate.py:934(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         9017 function calls in 0.092 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.092    0.092 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.092    0.092 surrogate.py:1721(__call__)
        1    0.002    0.002    0.088    0.088 surrogate.py:934(__call__)
        1    0.002    0.002    0.061    0.061 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.026    0.026 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.026    0.026    0.026    0.026 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
        1    0.021    0.021    0.021    0.021 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         3540 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.021    0.021 surrogate.py:934(__call__)
        1    0.001    0.001    0.015    0.015 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.006    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.006    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.006    0.000 surrogate.py:292(__call__)
        1    0.005    0.005    0.005    0.005 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.004    0.004    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
        2    0.000    0.000    0.004    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.004    0.002 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
       11    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      332    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         3536 function calls in 0.011 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.011    0.011 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.011    0.011 surrogate.py:1721(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:934(__call__)
        6    0.000    0.000    0.006    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.006    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.006    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         3540 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.038    0.038 surrogate.py:1721(__call__)
        1    0.000    0.000    0.033    0.033 surrogate.py:934(__call__)
        1    0.001    0.001    0.026    0.026 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.007    0.007 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.007    0.007    0.007    0.007 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.007    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.006    0.003    0.007    0.003 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.006    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.006    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.006    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
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
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         3523 function calls in 0.011 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.011    0.011 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.011    0.011 surrogate.py:1721(__call__)
        1    0.000    0.000    0.011    0.011 surrogate.py:934(__call__)
        1    0.000    0.000    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      332    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         3523 function calls in 0.164 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.164    0.164 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.012    0.012    0.164    0.164 surrogate.py:1721(__call__)
        1    0.000    0.000    0.144    0.144 surrogate.py:934(__call__)
        1    0.006    0.006    0.138    0.138 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.055    0.055    0.055    0.055 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.037    0.037 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.037    0.037    0.037    0.037 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.035    0.017 surrogate.py:91(_splinterp_Cwrapper)
        2    0.034    0.017    0.034    0.017 spline_interp_Cwrapper.py:50(interpolate)
        5    0.008    0.002    0.008    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.006    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.006    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1402(diff)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       20    0.000    0.000    0.000    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
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
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:155(__call__)
        1    0.000    0.000    0.003    0.003 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.001    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         3523 function calls in 0.039 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.039    0.039 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.039    0.039 surrogate.py:1721(__call__)
        1    0.000    0.000    0.036    0.036 surrogate.py:934(__call__)
        1    0.001    0.001    0.030    0.030 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.011    0.011    0.011    0.011 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.007    0.004    0.007    0.004 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         10972 function calls in 0.065 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.065    0.065 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.065    0.065 surrogate.py:1721(__call__)
        1    0.002    0.002    0.060    0.060 surrogate.py:934(__call__)
       11    0.000    0.000    0.030    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.030    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.029    0.002 surrogate.py:292(__call__)
        1    0.001    0.001    0.027    0.027 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.000    0.000    0.019    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.019    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.011    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.003 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 {method 'update' of 'dict' objects}
       18    0.003    0.000    0.003    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         10968 function calls in 0.042 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.042    0.042 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.042    0.042 surrogate.py:1721(__call__)
        1    0.002    0.002    0.042    0.042 surrogate.py:934(__call__)
       11    0.000    0.000    0.031    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.031    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.030    0.002 surrogate.py:292(__call__)
      218    0.000    0.000    0.020    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.019    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         10972 function calls in 0.095 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.095    0.095 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.095    0.095 surrogate.py:1721(__call__)
        1    0.002    0.002    0.086    0.086 surrogate.py:934(__call__)
        1    0.002    0.002    0.052    0.052 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.031    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.031    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.031    0.002 surrogate.py:292(__call__)
        1    0.000    0.000    0.021    0.021 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.021    0.021    0.021    0.021 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.020    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.020    0.000 nodeFunction.py:125(__call__)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.013    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.011    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.009    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.004    0.009    0.005 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.005    0.005 {method 'update' of 'dict' objects}
       18    0.005    0.000    0.005    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         10968 function calls in 0.043 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.043    0.043 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.043    0.043 surrogate.py:1721(__call__)
        1    0.002    0.002    0.042    0.042 surrogate.py:934(__call__)
       11    0.000    0.000    0.030    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.030    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.030    0.002 surrogate.py:292(__call__)
      218    0.000    0.000    0.020    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.019    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         10948 function calls in 0.043 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.043    0.043 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.043    0.043 surrogate.py:1721(__call__)
        1    0.002    0.002    0.043    0.043 surrogate.py:934(__call__)
       11    0.000    0.000    0.030    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.030    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.030    0.002 surrogate.py:292(__call__)
      218    0.000    0.000    0.020    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.019    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         10948 function calls in 0.300 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.300    0.300 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.024    0.024    0.300    0.300 surrogate.py:1721(__call__)
        1    0.004    0.004    0.264    0.264 surrogate.py:934(__call__)
        1    0.008    0.008    0.230    0.230 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.092    0.092 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.092    0.092    0.092    0.092 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.070    0.070    0.070    0.070 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.051    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.051    0.025    0.051    0.025 spline_interp_Cwrapper.py:50(interpolate)
       11    0.000    0.000    0.030    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.030    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.029    0.002 surrogate.py:292(__call__)
      218    0.000    0.000    0.019    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.019    0.000 nodeFunction.py:125(__call__)
        7    0.012    0.002    0.012    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.005    0.001    0.005    0.001 _function_base_impl.py:1402(diff)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         10948 function calls in 0.039 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.039    0.039 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.039    0.039 surrogate.py:1721(__call__)
        1    0.002    0.002    0.039    0.039 surrogate.py:934(__call__)
       11    0.000    0.000    0.029    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.029    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.029    0.002 surrogate.py:292(__call__)
      218    0.000    0.000    0.019    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.019    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.011    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         10948 function calls in 0.095 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.095    0.095 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.095    0.095 surrogate.py:1721(__call__)
        1    0.002    0.002    0.091    0.091 surrogate.py:934(__call__)
        1    0.002    0.002    0.058    0.058 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.029    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.029    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.029    0.002 surrogate.py:292(__call__)
        1    0.000    0.000    0.024    0.024 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.023    0.023    0.024    0.024 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.020    0.020    0.020    0.020 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.019    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.019    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.011    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

#### PR-73

##### NRSur7dq4 / mks_dt_0.000244140625_flow_0

```text
         26052 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:396(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:630(_integrate_forward)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:320(get_time_deriv)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:188(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:606(_one_backward_RK4_step)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:773(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:851(inertial_waveform_modes)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:810(_eval_comp)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:114(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:664(_integrate_backward)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:531(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      839    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:349(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:343(get_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         26053 function calls in 0.029 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.029    0.029 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:396(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:630(_integrate_forward)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:320(get_time_deriv)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:188(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:606(_one_backward_RK4_step)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:773(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:851(inertial_waveform_modes)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:810(_eval_comp)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:114(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:664(_integrate_backward)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:531(_initialize)
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      839    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:349(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:343(get_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         26052 function calls in 0.029 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.029    0.029 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:396(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:630(_integrate_forward)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:320(get_time_deriv)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:188(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:606(_one_backward_RK4_step)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:773(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:810(_eval_comp)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:114(rotateWaveform)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:664(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:531(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
      839    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:349(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         26053 function calls in 0.029 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.029    0.029 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:396(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:320(get_time_deriv)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:630(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:188(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:606(_one_backward_RK4_step)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:773(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:851(inertial_waveform_modes)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:810(_eval_comp)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:114(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      574    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:664(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:531(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      839    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:349(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:343(get_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         19278 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:396(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:630(_integrate_forward)
      238    0.001    0.000    0.007    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
        4    0.000    0.000    0.006    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:773(__call__)
        1    0.000    0.000    0.005    0.005 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.005    0.005    0.005    0.005 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:188(_eval_vector_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:810(_eval_comp)
     2390    0.001    0.000    0.004    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
     2390    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:851(inertial_waveform_modes)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:114(rotateWaveform)
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:565(_initial_RK4)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      716    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      534    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:712(_assemble_mode_pair)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:836(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:846(coorb_spins_from_copr_spins)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         26659 function calls in 0.027 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.027    0.027 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.027    0.027 surrogate.py:1721(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:396(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:320(get_time_deriv)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:188(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:606(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:773(__call__)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:810(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:630(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:664(_integrate_backward)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2876    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:851(inertial_waveform_modes)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:114(rotateWaveform)
      644    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:349(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:343(get_omega)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:531(_initialize)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
      862    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      839    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         19278 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1721(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:396(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:630(_integrate_forward)
      238    0.001    0.000    0.007    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:773(__call__)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:188(_eval_vector_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:810(_eval_comp)
     2390    0.001    0.000    0.004    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
     2390    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:851(inertial_waveform_modes)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:114(rotateWaveform)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:565(_initial_RK4)
      716    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      534    0.000    0.000    0.000    0.000 {built-in method numpy.array}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:712(_assemble_mode_pair)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:836(rotate_spin)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:846(coorb_spins_from_copr_spins)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         26659 function calls in 0.025 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.025    0.025 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.025    0.025 surrogate.py:1721(__call__)
        1    0.000    0.000    0.025    0.025 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:396(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:320(get_time_deriv)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:188(_eval_vector_fit)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:606(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:773(__call__)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:810(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:664(_integrate_backward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:630(_integrate_forward)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2876    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:851(inertial_waveform_modes)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:114(rotateWaveform)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:349(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:343(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:531(_initialize)
      862    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
      644    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      839    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         41113 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.037    0.037 surrogate.py:1721(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:1269(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:396(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:630(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:188(_eval_vector_fit)
     5135    0.003    0.000    0.012    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:320(get_time_deriv)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:606(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:664(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:114(rotateWaveform)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1195(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:773(__call__)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:810(_eval_comp)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:349(_get_t_from_omega)
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     1640    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:343(get_omega)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:531(_initialize)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      963    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        6    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         41114 function calls in 0.036 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.036    0.036 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.036    0.036 surrogate.py:1721(__call__)
        1    0.000    0.000    0.036    0.036 precessing_surrogate.py:1269(__call__)
        1    0.000    0.000    0.036    0.036 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:396(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:630(_integrate_forward)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:188(_eval_vector_fit)
     5135    0.003    0.000    0.012    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:320(get_time_deriv)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:606(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:664(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:114(rotateWaveform)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1195(<lambda>)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:773(__call__)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:810(_eval_comp)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:349(_get_t_from_omega)
     1640    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:343(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:531(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      963    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        6    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         41113 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1721(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:1269(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:396(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:630(_integrate_forward)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:188(_eval_vector_fit)
     5135    0.003    0.000    0.012    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:320(get_time_deriv)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:606(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:664(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:114(rotateWaveform)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1195(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:773(__call__)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:810(_eval_comp)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1640    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:349(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:343(get_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:531(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      963    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        6    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         41114 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.036    0.036 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.036    0.036 surrogate.py:1721(__call__)
        1    0.000    0.000    0.036    0.036 precessing_surrogate.py:1269(__call__)
        1    0.000    0.000    0.036    0.036 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:396(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:630(_integrate_forward)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:188(_eval_vector_fit)
     5135    0.003    0.000    0.012    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:320(get_time_deriv)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:606(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:664(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:114(rotateWaveform)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1195(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:773(__call__)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:810(_eval_comp)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1640    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:349(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:343(get_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:531(_initialize)
      963    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        6    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         33092 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 90 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.037    0.037 surrogate.py:1721(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:1269(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:396(__call__)
        1    0.001    0.001    0.020    0.020 precessing_surrogate.py:630(_integrate_forward)
      505    0.001    0.000    0.018    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
     1515    0.003    0.000    0.014    0.000 precessing_surrogate.py:188(_eval_vector_fit)
     4637    0.003    0.000    0.011    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
        4    0.000    0.000    0.010    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.009    0.009 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.009    0.009 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:114(rotateWaveform)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1195(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:773(__call__)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:810(_eval_comp)
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     1517    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        6    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:565(_initial_RK4)
      623    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:836(rotate_spin)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         42469 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1721(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:1269(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:396(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:188(_eval_vector_fit)
     5274    0.003    0.000    0.013    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:664(_integrate_backward)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:630(_integrate_forward)
     5274    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:320(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:606(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:114(rotateWaveform)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1195(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:773(__call__)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:810(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:349(_get_t_from_omega)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:343(get_omega)
      906    0.001    0.000    0.002    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     1640    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:531(_initialize)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     1102    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        6    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         33092 function calls in 0.030 seconds

   Ordered by: cumulative time
   List reduced from 90 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.030    0.030 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.030    0.030 surrogate.py:1721(__call__)
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:1269(__call__)
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:396(__call__)
        1    0.001    0.001    0.019    0.019 precessing_surrogate.py:630(_integrate_forward)
      505    0.001    0.000    0.017    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
     1515    0.003    0.000    0.013    0.000 precessing_surrogate.py:188(_eval_vector_fit)
     4637    0.003    0.000    0.011    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
     4637    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:114(rotateWaveform)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1195(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:773(__call__)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:810(_eval_comp)
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     1517    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        6    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
      597    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:565(_initial_RK4)
      623    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:836(rotate_spin)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:846(coorb_spins_from_copr_spins)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         42469 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.037    0.037 surrogate.py:1721(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:1269(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:941(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:396(__call__)
      546    0.002    0.000    0.020    0.000 precessing_surrogate.py:302(get_time_deriv_from_index)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:188(_eval_vector_fit)
     5274    0.003    0.000    0.013    0.000 precessing_surrogate.py:161(_eval_scalar_fit)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:664(_integrate_backward)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:630(_integrate_forward)
     5274    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:320(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:606(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:114(rotateWaveform)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1195(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:773(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:810(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:349(_get_t_from_omega)
      906    0.001    0.000    0.002    0.000 surrogate.py:2578(get_fit_params)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:343(get_omega)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1640    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:531(_initialize)
     1102    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        6    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         9044 function calls in 0.064 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.064    0.064 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.064    0.064 surrogate.py:1721(__call__)
        1    0.004    0.004    0.059    0.059 surrogate.py:934(__call__)
        1    0.001    0.001    0.032    0.032 surrogate.py:742(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        1    0.000    0.000    0.014    0.014 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.013    0.013    0.014    0.014 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.003    0.005    0.003 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       21    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1402(diff)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        9    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 {built-in method numpy.ascontiguousarray}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         9040 function calls in 0.033 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.033    0.033 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.033    0.033 surrogate.py:1721(__call__)
        1    0.002    0.002    0.032    0.032 surrogate.py:934(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         9044 function calls in 0.106 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.106    0.106 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.106    0.106 surrogate.py:1721(__call__)
        1    0.004    0.004    0.086    0.086 surrogate.py:934(__call__)
        1    0.003    0.003    0.058    0.058 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.024    0.024 surrogate.py:86(_splinterp_Cwrapper_many_complex)
       12    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
        1    0.023    0.023    0.024    0.024 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.016    0.016 {method 'update' of 'dict' objects}
       21    0.016    0.001    0.016    0.001 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        4    0.003    0.001    0.003    0.001 _function_base_impl.py:1402(diff)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         9040 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1721(__call__)
        1    0.002    0.002    0.034    0.034 surrogate.py:934(__call__)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
       21    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         9017 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1721(__call__)
        1    0.002    0.002    0.034    0.034 surrogate.py:934(__call__)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         9017 function calls in 0.314 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.314    0.314 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.028    0.028    0.314    0.314 surrogate.py:1721(__call__)
        1    0.003    0.003    0.273    0.273 surrogate.py:934(__call__)
        1    0.010    0.010    0.245    0.245 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.101    0.101 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.101    0.101    0.101    0.101 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.074    0.074    0.074    0.074 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.052    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.052    0.026    0.052    0.026 spline_interp_Cwrapper.py:50(interpolate)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
        9    0.013    0.001    0.013    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         9017 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1721(__call__)
        1    0.004    0.004    0.034    0.034 surrogate.py:934(__call__)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        9    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         9017 function calls in 0.091 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.091    0.091 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.091    0.091 surrogate.py:1721(__call__)
        1    0.002    0.002    0.086    0.086 surrogate.py:934(__call__)
        1    0.002    0.002    0.060    0.060 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.026    0.026 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.026    0.026    0.026    0.026 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
        1    0.020    0.020    0.020    0.020 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
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
        1    0.001    0.001    0.015    0.015 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
        1    0.005    0.005    0.005    0.005 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.004    0.004    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
        2    0.000    0.000    0.004    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.004    0.002 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
       11    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
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
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
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
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         3540 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.037    0.037 surrogate.py:1721(__call__)
        1    0.000    0.000    0.032    0.032 surrogate.py:934(__call__)
        1    0.001    0.001    0.026    0.026 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.007    0.007 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.007    0.007    0.007    0.007 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.007    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.006    0.003    0.007    0.003 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.006    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.006    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.006    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
        1    0.000    0.000    0.003    0.003 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
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
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         3523 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1721(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:934(__call__)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
        1    0.000    0.000    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         3523 function calls in 0.164 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.164    0.164 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.012    0.012    0.164    0.164 surrogate.py:1721(__call__)
        1    0.000    0.000    0.144    0.144 surrogate.py:934(__call__)
        1    0.006    0.006    0.138    0.138 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.055    0.055    0.055    0.055 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.037    0.037 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.037    0.037    0.037    0.037 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.035    0.017 surrogate.py:91(_splinterp_Cwrapper)
        2    0.034    0.017    0.035    0.017 spline_interp_Cwrapper.py:50(interpolate)
        5    0.008    0.002    0.008    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1402(diff)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 enum.py:187(__get__)
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
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
        1    0.000    0.000    0.003    0.003 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.001    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         3523 function calls in 0.039 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.039    0.039 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.039    0.039 surrogate.py:1721(__call__)
        1    0.000    0.000    0.037    0.037 surrogate.py:934(__call__)
        1    0.001    0.001    0.030    0.030 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.011    0.011    0.011    0.011 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.007    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.006    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         10972 function calls in 0.063 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.063    0.063 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.063    0.063 surrogate.py:1721(__call__)
        1    0.002    0.002    0.058    0.058 surrogate.py:934(__call__)
       11    0.000    0.000    0.029    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.029    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.028    0.001 surrogate.py:292(__call__)
        1    0.001    0.001    0.026    0.026 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.000    0.000    0.019    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.019    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 {method 'update' of 'dict' objects}
       18    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         10968 function calls in 0.040 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.040    0.040 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.040    0.040 surrogate.py:1721(__call__)
        1    0.002    0.002    0.039    0.039 surrogate.py:934(__call__)
       11    0.000    0.000    0.030    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.030    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.029    0.002 surrogate.py:292(__call__)
      218    0.000    0.000    0.020    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.020    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         10972 function calls in 0.093 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.093    0.093 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.093    0.093 surrogate.py:1721(__call__)
        1    0.002    0.002    0.085    0.085 surrogate.py:934(__call__)
        1    0.002    0.002    0.051    0.051 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.030    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.030    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.030    0.002 surrogate.py:292(__call__)
        1    0.000    0.000    0.022    0.022 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.021    0.021    0.021    0.021 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.020    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.020    0.000 nodeFunction.py:125(__call__)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.009    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.004    0.009    0.005 spline_interp_Cwrapper.py:50(interpolate)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.005    0.005 {method 'update' of 'dict' objects}
       18    0.005    0.000    0.005    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         10968 function calls in 0.044 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.044    0.044 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.044    0.044 surrogate.py:1721(__call__)
        1    0.002    0.002    0.043    0.043 surrogate.py:934(__call__)
       11    0.000    0.000    0.031    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.031    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.030    0.002 surrogate.py:292(__call__)
      218    0.000    0.000    0.020    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.020    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         10948 function calls in 0.041 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.041    0.041 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.041    0.041 surrogate.py:1721(__call__)
        1    0.002    0.002    0.040    0.040 surrogate.py:934(__call__)
       11    0.000    0.000    0.029    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.029    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.029    0.002 surrogate.py:292(__call__)
      218    0.000    0.000    0.020    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.019    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         10948 function calls in 0.295 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.295    0.295 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.021    0.021    0.295    0.295 surrogate.py:1721(__call__)
        1    0.004    0.004    0.261    0.261 surrogate.py:934(__call__)
        1    0.007    0.007    0.228    0.228 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.093    0.093 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.092    0.092    0.093    0.093 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.069    0.069    0.069    0.069 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.050    0.025 surrogate.py:91(_splinterp_Cwrapper)
        2    0.049    0.025    0.050    0.025 spline_interp_Cwrapper.py:50(interpolate)
       11    0.000    0.000    0.029    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.029    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.029    0.002 surrogate.py:292(__call__)
      218    0.000    0.000    0.019    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.019    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        7    0.012    0.002    0.012    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.010    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.005    0.001    0.005    0.001 _function_base_impl.py:1402(diff)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         10948 function calls in 0.039 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.039    0.039 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.039    0.039 surrogate.py:1721(__call__)
        1    0.002    0.002    0.039    0.039 surrogate.py:934(__call__)
       11    0.000    0.000    0.030    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.030    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.030    0.002 surrogate.py:292(__call__)
      218    0.000    0.000    0.020    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.020    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         10948 function calls in 0.097 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.097    0.097 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.097    0.097 surrogate.py:1721(__call__)
        1    0.002    0.002    0.093    0.093 surrogate.py:934(__call__)
        1    0.002    0.002    0.059    0.059 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.030    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.030    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.030    0.002 surrogate.py:292(__call__)
        1    0.000    0.000    0.025    0.025 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.024    0.024    0.025    0.025 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.020    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.020    0.000 nodeFunction.py:125(__call__)
        1    0.019    0.019    0.019    0.019 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.013    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.010    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.011    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```
