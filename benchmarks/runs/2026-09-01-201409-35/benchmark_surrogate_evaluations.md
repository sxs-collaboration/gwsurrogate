# GWSurrogate Evaluation Timing

Generated: 2026-09-01T20:14:02.952581+00:00

Times below are seconds per model evaluation. Raw repeats and context are in the JSON output.

PNG timing table: `test/benchmark_surrogate_evaluations.png`

## Summary

### master

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0144811` s, median `0.0146292` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0142594` s, median `0.014415` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0152543` s, median `0.0155136` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0151193` s, median `0.0152615` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0121667` s, median `0.0123393` s
- `dt=0.1 M`, `f_low=0.01`: best `0.011859` s, median `0.0119318` s
- `dt=0.5 M`, `f_low=0`: best `0.00832712` s, median `0.00849698` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0110715` s, median `0.0111615` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0158168` s, median `0.016123` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0151438` s, median `0.0152185` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0169637` s, median `0.0171382` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0158673` s, median `0.0159402` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0192878` s, median `0.0194249` s
- `dt=0.1 M`, `f_low=0.01`: best `0.016367` s, median `0.0166518` s
- `dt=0.5 M`, `f_low=0`: best `0.0126786` s, median `0.0127621` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0151401` s, median `0.0152273` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0600236` s, median `0.060633` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.02753` s, median `0.0281332` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.101643` s, median `0.102071` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0290282` s, median `0.0292093` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0293166` s, median `0.0296034` s
- `dt=0.1 M`, `f_low=0.002`: best `0.303954` s, median `0.304811` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0265365` s, median `0.0266685` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0850679` s, median `0.0856529` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0206307` s, median `0.0207381` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.00713845` s, median `0.00722462` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0342827` s, median `0.0345346` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.00807067` s, median `0.00813814` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0082261` s, median `0.00832198` s
- `dt=0.1 M`, `f_low=0.002`: best `0.161604` s, median `0.161907` s
- `dt=0.5 M`, `f_low=0.01`: best `0.00648868` s, median `0.00672338` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0361662` s, median `0.0362847` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0538821` s, median `0.0541182` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0303584` s, median `0.0305582` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.080698` s, median `0.0811713` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0319836` s, median `0.0325297` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0323933` s, median `0.0329878` s
- `dt=0.1 M`, `f_low=0.002`: best `0.287304` s, median `0.287888` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0292228` s, median `0.0295956` s
- `dt=0.5 M`, `f_low=0.002`: best `0.084846` s, median `0.0859854` s

### PR-67

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0143411` s, median `0.014715` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0142465` s, median `0.0145704` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0154204` s, median `0.0155067` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.014724` s, median `0.0148392` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0121572` s, median `0.0122904` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0118443` s, median `0.0120915` s
- `dt=0.5 M`, `f_low=0`: best `0.00814836` s, median `0.0082929` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0109847` s, median `0.0112377` s

#### NRSur7dq4v2

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0227697` s, median `0.0228688` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0224674` s, median `0.0227542` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0235976` s, median `0.0237108` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.02305` s, median `0.0232709` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0339153` s, median `0.0342161` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0322033` s, median `0.0324897` s
- `dt=0.5 M`, `f_low=0`: best `0.0282424` s, median `0.0285104` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0307887` s, median `0.0308423` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0133673` s, median `0.0134885` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.013368` s, median `0.013855` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.014395` s, median `0.0147397` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.013788` s, median `0.0139246` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0193029` s, median `0.0194623` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0164934` s, median `0.0167532` s
- `dt=0.5 M`, `f_low=0`: best `0.0126202` s, median `0.0127685` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0151818` s, median `0.0153872` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0593458` s, median `0.0594834` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0277333` s, median `0.0280441` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.101797` s, median `0.102412` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.029178` s, median `0.0293032` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0295142` s, median `0.0297494` s
- `dt=0.1 M`, `f_low=0.002`: best `0.302318` s, median `0.302867` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0265272` s, median `0.0266176` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0851979` s, median `0.0854309` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0208573` s, median `0.0210586` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.00729568` s, median `0.00738196` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0348196` s, median `0.0350385` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.00822482` s, median `0.00853255` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.00840986` s, median `0.00846003` s
- `dt=0.1 M`, `f_low=0.002`: best `0.162208` s, median `0.162526` s
- `dt=0.5 M`, `f_low=0.01`: best `0.00652311` s, median `0.00660315` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0361584` s, median `0.0362817` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0536718` s, median `0.0541526` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0307052` s, median `0.0311831` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0905919` s, median `0.0910608` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0322899` s, median `0.0324295` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0324002` s, median `0.0327732` s
- `dt=0.1 M`, `f_low=0.002`: best `0.287134` s, median `0.287925` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0295511` s, median `0.0300786` s
- `dt=0.5 M`, `f_low=0.002`: best `0.084355` s, median `0.0850013` s

## Context

### master

- Git branch: `master`
- Git commit: `5d076870ef24cadd7417cbbba91596a233c820c3`
- Git describe: `v1.1.9-2-g5d07687`
- Python: `3.14.7 (main, Aug  6 2026, 02:19:46) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1022-azure x86_64`
- CPU count: `4`
- Conda env: `unknown`

Submodules:

- `gwsurrogate/eval_pysur`: `6a51ecba0ed8ddc26a85e5d2918596aa9f58f534` initialized ((heads/master))

### PR-67

- Git branch: `unknown`
- Git commit: `6f539427539d021ba142b3a5a7cf9b96c83e09c4`
- Git describe: `v1.1.9-55-g6f53942`
- Python: `3.14.7 (main, Aug  6 2026, 02:19:46) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1022-azure x86_64`
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

#### PR-67

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
         9221 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1721(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      750    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
       22    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         9222 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1721(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:754(__call__)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
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
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      254    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         9221 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1721(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      750    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         9222 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1721(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:754(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      750    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         4175 function calls in 0.014 seconds

   Ordered by: cumulative time
   List reduced from 85 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.014    0.014 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.014    0.014 surrogate.py:1721(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:940(__call__)
        4    0.000    0.000    0.006    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.005    0.005 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.005    0.005    0.005    0.005 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
      238    0.000    0.000    0.002    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
      486    0.001    0.000    0.001    0.000 surrogate.py:2446(get_fit_params)
      238    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        3    0.000    0.000    0.000    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      492    0.000    0.000    0.000    0.000 {built-in method numpy.array}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:844(coorb_spins_from_copr_spins)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:868(normalize_spin)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
       13    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:112(<genexpr>)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         9758 function calls in 0.016 seconds

   Ordered by: cumulative time
   List reduced from 97 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.016    0.016 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.016    0.016 surrogate.py:1721(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
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
      820    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      650    0.000    0.000    0.000    0.000 __init__.py:271(POINTER)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         4175 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 85 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1721(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
      238    0.000    0.000    0.002    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      486    0.001    0.000    0.001    0.000 surrogate.py:2446(get_fit_params)
      238    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        3    0.000    0.000    0.000    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      492    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:844(coorb_spins_from_copr_spins)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:868(normalize_spin)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
       13    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:112(<genexpr>)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         9758 function calls in 0.015 seconds

   Ordered by: cumulative time
   List reduced from 97 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.015    0.015 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.015    0.015 surrogate.py:1721(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:754(__call__)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:791(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      820    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         11494 function calls in 0.020 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.020    0.020 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.020    0.020 surrogate.py:1721(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:607(_integrate_forward)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      546    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:851(inertial_waveform_modes)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
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
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
      943    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         11495 function calls in 0.020 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.020    0.020 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1721(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:607(_integrate_forward)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      546    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
      943    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         11494 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:297(get_time_deriv)
      546    0.000    0.000    0.006    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:607(_integrate_forward)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.005    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      546    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:508(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
      943    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         11495 function calls in 0.020 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.020    0.020 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.020    0.020 surrogate.py:1721(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:607(_integrate_forward)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      546    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:851(inertial_waveform_modes)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
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
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
      943    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         5781 function calls in 0.021 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.021    0.021 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.021    0.021 surrogate.py:1721(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:940(__call__)
        4    0.000    0.000    0.010    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.009    0.009 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.009    0.009 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:607(_integrate_forward)
      505    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
      505    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      603    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:844(coorb_spins_from_copr_spins)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:868(normalize_spin)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         12572 function calls in 0.021 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.021    0.021 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.021    0.021 surrogate.py:1721(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      546    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:607(_integrate_forward)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     1082    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      360    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      268    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         5781 function calls in 0.015 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.015    0.015 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.015    0.015 surrogate.py:1721(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:607(_integrate_forward)
      505    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      505    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
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
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:844(coorb_spins_from_copr_spins)
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:689(_assemble_mode_pair)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:868(normalize_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         12572 function calls in 0.020 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.020    0.020 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.020    0.020 surrogate.py:1721(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:1270(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:940(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      546    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:851(inertial_waveform_modes)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:641(_integrate_backward)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:754(__call__)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:791(_eval_comp)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:858(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     1082    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      268    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      360    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         9044 function calls in 0.064 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.064    0.064 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.064    0.064 surrogate.py:1721(__call__)
        1    0.003    0.003    0.059    0.059 surrogate.py:934(__call__)
        1    0.001    0.001    0.032    0.032 surrogate.py:742(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
        1    0.000    0.000    0.015    0.015 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.014    0.014    0.015    0.015 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
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
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1413(diff)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 {built-in method numpy.ascontiguousarray}
        9    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         9040 function calls in 0.033 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.033    0.033 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.033    0.033 surrogate.py:1721(__call__)
        1    0.002    0.002    0.032    0.032 surrogate.py:934(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         9044 function calls in 0.102 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.102    0.102 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.102    0.102 surrogate.py:1721(__call__)
        1    0.003    0.003    0.083    0.083 surrogate.py:934(__call__)
        1    0.002    0.002    0.056    0.056 surrogate.py:742(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.015    0.015 {method 'update' of 'dict' objects}
       21    0.015    0.001    0.015    0.001 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        4    0.003    0.001    0.003    0.001 _function_base_impl.py:1413(diff)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
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
         9040 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1721(__call__)
        1    0.002    0.002    0.033    0.033 surrogate.py:934(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
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
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
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
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         9017 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1721(__call__)
        1    0.002    0.002    0.034    0.034 surrogate.py:934(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         9017 function calls in 0.308 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.308    0.308 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.027    0.027    0.308    0.308 surrogate.py:1721(__call__)
        1    0.003    0.003    0.269    0.269 surrogate.py:934(__call__)
        1    0.009    0.009    0.243    0.243 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.100    0.100 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.100    0.100    0.100    0.100 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.073    0.073    0.073    0.073 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.053    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.052    0.026    0.053    0.026 spline_interp_Cwrapper.py:50(interpolate)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.014    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
        9    0.012    0.001    0.012    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1413(diff)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5592(append)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        6    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         9017 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1721(__call__)
        1    0.002    0.002    0.031    0.031 surrogate.py:934(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         9017 function calls in 0.090 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.090    0.090 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.090    0.090 surrogate.py:1721(__call__)
        1    0.002    0.002    0.086    0.086 surrogate.py:934(__call__)
        1    0.002    0.002    0.060    0.060 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.026    0.026 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.026    0.026    0.026    0.026 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
        1    0.020    0.020    0.020    0.020 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         3540 function calls in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.022    0.022 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.022    0.022 surrogate.py:1721(__call__)
        1    0.000    0.000    0.020    0.020 surrogate.py:934(__call__)
        1    0.001    0.001    0.014    0.014 surrogate.py:742(_coorbital_to_inertial_frame)
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
        3    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         3536 function calls in 0.009 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.009    0.009 benchmark_surrogate_evaluations.py:355(evaluate_case)
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
        2    0.000    0.000    0.001    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 enum.py:187(__get__)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         3540 function calls in 0.036 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.036    0.036 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.036    0.036 surrogate.py:1721(__call__)
        1    0.000    0.000    0.032    0.032 surrogate.py:934(__call__)
        1    0.001    0.001    0.026    0.026 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.007    0.007 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.007    0.007    0.007    0.007 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.007    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.006    0.003    0.007    0.003 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.006    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.006    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         3536 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1721(__call__)
        1    0.000    0.000    0.009    0.009 surrogate.py:934(__call__)
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
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         3523 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1721(__call__)
        1    0.000    0.000    0.009    0.009 surrogate.py:934(__call__)
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
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       20    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         3523 function calls in 0.163 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.163    0.163 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.012    0.012    0.163    0.163 surrogate.py:1721(__call__)
        1    0.000    0.000    0.143    0.143 surrogate.py:934(__call__)
        1    0.005    0.005    0.137    0.137 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.056    0.056    0.056    0.056 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.037    0.037 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.037    0.037    0.037    0.037 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.034    0.017 surrogate.py:91(_splinterp_Cwrapper)
        2    0.034    0.017    0.034    0.017 spline_interp_Cwrapper.py:50(interpolate)
        5    0.007    0.001    0.007    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.006    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.006    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.006    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.001    0.000    0.002    0.000 _function_base_impl.py:1413(diff)
        2    0.001    0.001    0.001    0.001 surrogate.py:732(_search_omega)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         3523 function calls in 0.008 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.008    0.008 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.008    0.008 surrogate.py:1721(__call__)
        1    0.000    0.000    0.008    0.008 surrogate.py:934(__call__)
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
        2    0.000    0.000    0.001    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      332    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         3523 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.038    0.038 surrogate.py:1721(__call__)
        1    0.000    0.000    0.035    0.035 surrogate.py:934(__call__)
        1    0.001    0.001    0.029    0.029 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.011    0.011    0.011    0.011 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.007    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         10972 function calls in 0.061 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.061    0.061 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.061    0.061 surrogate.py:1721(__call__)
        1    0.002    0.002    0.057    0.057 surrogate.py:934(__call__)
       11    0.000    0.000    0.028    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.028    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.027    0.001 surrogate.py:292(__call__)
        1    0.001    0.001    0.026    0.026 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.000    0.000    0.019    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.019    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.011    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.003 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.004    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       18    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         10968 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1721(__call__)
        1    0.002    0.002    0.038    0.038 surrogate.py:934(__call__)
       11    0.000    0.000    0.029    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.029    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.028    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.020    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.020    0.000 nodeFunction.py:125(__call__)
      218    0.002    0.000    0.013    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.011    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         10972 function calls in 0.087 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.087    0.087 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.087    0.087 surrogate.py:1721(__call__)
        1    0.002    0.002    0.080    0.080 surrogate.py:934(__call__)
        1    0.002    0.002    0.050    0.050 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.027    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.021    0.021 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.020    0.020    0.021    0.021 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
        1    0.016    0.016    0.016    0.016 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.011    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.009    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.004    0.009    0.005 spline_interp_Cwrapper.py:50(interpolate)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       18    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         10968 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1721(__call__)
        1    0.002    0.002    0.037    0.037 surrogate.py:934(__call__)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         10948 function calls in 0.039 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.039    0.039 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.039    0.039 surrogate.py:1721(__call__)
        1    0.002    0.002    0.038    0.038 surrogate.py:934(__call__)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.027    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.011    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      654    0.000    0.000    0.000    0.000 {built-in method _warnings._acquire_lock}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         10948 function calls in 0.292 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.292    0.292 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.021    0.021    0.292    0.292 surrogate.py:1721(__call__)
        1    0.003    0.003    0.259    0.259 surrogate.py:934(__call__)
        1    0.007    0.007    0.228    0.228 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.092    0.092 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.091    0.091    0.092    0.092 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.070    0.070    0.070    0.070 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.052    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.052    0.026    0.052    0.026 spline_interp_Cwrapper.py:50(interpolate)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.027    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.011    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        7    0.011    0.002    0.011    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.009    0.000    0.011    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1413(diff)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.001    0.000 _function_base_impl.py:5592(append)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         10948 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1721(__call__)
        1    0.002    0.002    0.035    0.035 surrogate.py:934(__call__)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
      654    0.000    0.000    0.000    0.000 {built-in method _warnings._acquire_lock}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         10948 function calls in 0.090 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.090    0.090 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.090    0.090 surrogate.py:1721(__call__)
        1    0.002    0.002    0.087    0.087 surrogate.py:934(__call__)
        1    0.002    0.002    0.057    0.057 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.027    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.024    0.024 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.023    0.023    0.024    0.024 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.019    0.019    0.019    0.019 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.019    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.011    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
      218    0.009    0.000    0.011    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.010    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

#### PR-67

##### NRSur7dq4 / mks_dt_0.000244140625_flow_0

```text
         9264 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1722(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:42(_wignerD_matrices)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      574    0.001    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      750    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         9265 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1722(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      574    0.001    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      750    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         9264 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1722(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      574    0.001    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      750    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         9265 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1722(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:373(__call__)
       13    0.001    0.000    0.008    0.001 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.007    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.006    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.003    0.000    0.006    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
      279    0.000    0.000    0.004    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:875(__call__)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      574    0.002    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.002    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
      646    0.001    0.000    0.001    0.000 _internal.py:262(__init__)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      750    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      650    0.000    0.000    0.000    0.000 __init__.py:271(POINTER)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         4218 function calls in 0.014 seconds

   Ordered by: cumulative time
   List reduced from 86 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.014    0.014 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.014    0.014 surrogate.py:1722(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:1214(__call__)
        4    0.000    0.000    0.006    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.005    0.005 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.005    0.005    0.005    0.005 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
      238    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      486    0.001    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
      238    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        3    0.000    0.000    0.000    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      492    0.000    0.000    0.000    0.000 {built-in method numpy.array}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1101(coorb_spins_from_copr_spins)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         9801 function calls in 0.016 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.016    0.016 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.016    0.016 surrogate.py:1722(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
      644    0.001    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      820    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         4218 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 86 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1722(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
      238    0.000    0.000    0.002    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      486    0.001    0.000    0.001    0.000 surrogate.py:2451(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      238    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        3    0.000    0.000    0.000    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      492    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1101(coorb_spins_from_copr_spins)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
       13    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:112(<genexpr>)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         9801 function calls in 0.016 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.016    0.016 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.016    0.016 surrogate.py:1722(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
      644    0.001    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      820    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
```

##### NRSur7dq4v2 / mks_dt_0.000244140625_flow_0

```text
         20275 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1722(__call__)
        1    0.000    0.000    0.032    0.032 precessing_surrogate.py:1214(__call__)
        1    0.001    0.001    0.018    0.018 precessing_surrogate.py:1015(__call__)
       84    0.000    0.000    0.017    0.000 precessing_surrogate.py:1061(_eval_comp)
       84    0.004    0.000    0.017    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     2264    0.008    0.000    0.011    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      279    0.000    0.000    0.004    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
     4704    0.003    0.000    0.003    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
     1985    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
     2264    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       86    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:508(_initialize)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       18    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
     2264    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
```

##### NRSur7dq4v2 / mks_dt_0.000244140625_flow_20

```text
         20276 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1722(__call__)
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1015(__call__)
       84    0.000    0.000    0.016    0.000 precessing_surrogate.py:1061(_eval_comp)
       84    0.004    0.000    0.016    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     2264    0.007    0.000    0.011    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
     4704    0.003    0.000    0.003    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
     1985    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
     2264    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
       86    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       18    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
     2264    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      356    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4v2 / mks_dt_0.0001220703125_flow_0

```text
         20275 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1722(__call__)
        1    0.000    0.000    0.032    0.032 precessing_surrogate.py:1214(__call__)
        1    0.001    0.001    0.017    0.017 precessing_surrogate.py:1015(__call__)
       84    0.000    0.000    0.017    0.000 precessing_surrogate.py:1061(_eval_comp)
       84    0.004    0.000    0.017    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     2264    0.007    0.000    0.011    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
     4704    0.003    0.000    0.003    0.000 {built-in method numpy.array}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
     1985    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
     2264    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
       86    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       18    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
     2264    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRSur7dq4v2 / mks_dt_0.0001220703125_flow_20

```text
         20276 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1722(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1015(__call__)
       84    0.000    0.000    0.016    0.000 precessing_surrogate.py:1061(_eval_comp)
       84    0.004    0.000    0.016    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     2264    0.007    0.000    0.011    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
     4704    0.003    0.000    0.003    0.000 {built-in method numpy.array}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
     1985    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     2264    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
       86    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:326(_get_t_from_omega)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       18    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
     2264    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
```

##### NRSur7dq4v2 / geom_dt_0.1_flow_0

```text
         21680 function calls in 0.043 seconds

   Ordered by: cumulative time
   List reduced from 86 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.043    0.043 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.043    0.043 surrogate.py:1722(__call__)
        1    0.000    0.000    0.043    0.043 precessing_surrogate.py:1214(__call__)
        1    0.001    0.001    0.026    0.026 precessing_surrogate.py:1015(__call__)
      128    0.000    0.000    0.025    0.000 precessing_surrogate.py:1061(_eval_comp)
      128    0.006    0.000    0.025    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     3195    0.010    0.000    0.015    0.000 surrogate.py:2564(get_fit_params)
        4    0.000    0.000    0.008    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:607(_integrate_forward)
     6396    0.004    0.000    0.004    0.000 {built-in method numpy.array}
      238    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
     2957    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
     3195    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
      130    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
      238    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        3    0.000    0.000    0.000    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
       28    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
     3195    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      322    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1101(coorb_spins_from_copr_spins)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
```

##### NRSur7dq4v2 / geom_dt_0.1_flow_0.01

```text
         27421 function calls in 0.043 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.043    0.043 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.043    0.043 surrogate.py:1722(__call__)
        1    0.000    0.000    0.043    0.043 precessing_surrogate.py:1214(__call__)
        1    0.001    0.001    0.026    0.026 precessing_surrogate.py:1015(__call__)
      128    0.000    0.000    0.025    0.000 precessing_surrogate.py:1061(_eval_comp)
      128    0.006    0.000    0.025    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     3353    0.011    0.000    0.015    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     6882    0.004    0.000    0.004    0.000 {built-in method numpy.array}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     3074    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     3353    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      130    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
       28    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
     3353    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      465    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
```

##### NRSur7dq4v2 / geom_dt_0.5_flow_0

```text
         21680 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 86 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.037    0.037 surrogate.py:1722(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:1214(__call__)
        1    0.001    0.001    0.026    0.026 precessing_surrogate.py:1015(__call__)
      128    0.000    0.000    0.025    0.000 precessing_surrogate.py:1061(_eval_comp)
      128    0.006    0.000    0.025    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     3195    0.010    0.000    0.015    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:607(_integrate_forward)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
     6396    0.004    0.000    0.004    0.000 {built-in method numpy.array}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      238    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
     2957    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
     3195    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
      130    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
      238    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
       28    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
     3195    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      322    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1101(coorb_spins_from_copr_spins)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
```

##### NRSur7dq4v2 / geom_dt_0.5_flow_0.01

```text
         27421 function calls in 0.041 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.041    0.041 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.041    0.041 surrogate.py:1722(__call__)
        1    0.000    0.000    0.041    0.041 precessing_surrogate.py:1214(__call__)
        1    0.001    0.001    0.026    0.026 precessing_surrogate.py:1015(__call__)
      128    0.000    0.000    0.025    0.000 precessing_surrogate.py:1061(_eval_comp)
      128    0.006    0.000    0.025    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     3353    0.011    0.000    0.016    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     6882    0.004    0.000    0.004    0.000 {built-in method numpy.array}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
     3074    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     3353    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      130    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
       28    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
     3353    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      465    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         11279 function calls in 0.017 seconds

   Ordered by: cumulative time
   List reduced from 102 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.017    0.017 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.017    0.017 surrogate.py:1722(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1557(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:607(_integrate_forward)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      546    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      750    0.001    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:912(_eval_comp)
      750    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      926    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      204    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         11280 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 102 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1722(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:1557(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:607(_integrate_forward)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      546    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      750    0.001    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:912(_eval_comp)
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      750    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
      926    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      204    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         11279 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 102 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1722(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:1557(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:607(_integrate_forward)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      546    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
      750    0.001    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      750    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
       16    0.000    0.000    0.000    0.000 precessing_surrogate.py:912(_eval_comp)
       16    0.000    0.000    0.000    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      926    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      204    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         11280 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 102 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.017    0.017 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.017    0.017 surrogate.py:1722(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1557(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:607(_integrate_forward)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
      546    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      750    0.000    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      750    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:912(_eval_comp)
       16    0.000    0.000    0.000    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      926    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      204    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         5802 function calls in 0.021 seconds

   Ordered by: cumulative time
   List reduced from 88 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.021    0.021 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.021    0.021 surrogate.py:1722(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:1557(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:1214(__call__)
        4    0.000    0.000    0.010    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.009    0.009 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.009    0.009 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:607(_integrate_forward)
      505    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      505    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      597    0.000    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:912(_eval_comp)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      603    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1101(coorb_spins_from_copr_spins)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         12593 function calls in 0.021 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.021    0.021 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.021    0.021 surrogate.py:1722(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:1557(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:373(__call__)
      546    0.001    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      546    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      906    0.001    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:912(_eval_comp)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     1082    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      268    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      360    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         5802 function calls in 0.015 seconds

   Ordered by: cumulative time
   List reduced from 88 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.015    0.015 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.015    0.015 surrogate.py:1722(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:1557(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:607(_integrate_forward)
      505    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
      505    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      597    0.000    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:912(_eval_comp)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      603    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1101(coorb_spins_from_copr_spins)
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         12593 function calls in 0.020 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.020    0.020 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.020    0.020 surrogate.py:1722(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:1557(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      546    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1108(inertial_waveform_modes)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:641(_integrate_backward)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      906    0.001    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:912(_eval_comp)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     1082    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      268    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      360    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         9044 function calls in 0.063 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.063    0.063 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.063    0.063 surrogate.py:1722(__call__)
        1    0.003    0.003    0.058    0.058 surrogate.py:934(__call__)
        1    0.001    0.001    0.031    0.031 surrogate.py:742(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        1    0.000    0.000    0.014    0.014 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.014    0.014    0.014    0.014 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.003 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 {method 'update' of 'dict' objects}
       21    0.003    0.000    0.003    0.000 surrogate.py:2131(<genexpr>)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        9    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        3    0.001    0.000    0.001    0.000 {built-in method numpy.ascontiguousarray}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         9040 function calls in 0.033 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.033    0.033 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.033    0.033 surrogate.py:1722(__call__)
        1    0.002    0.002    0.032    0.032 surrogate.py:934(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
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
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       21    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         9044 function calls in 0.104 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.104    0.104 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.104    0.104 surrogate.py:1722(__call__)
        1    0.003    0.003    0.085    0.085 surrogate.py:934(__call__)
        1    0.002    0.002    0.058    0.058 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.024    0.024 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.023    0.023    0.024    0.024 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.015    0.015 {method 'update' of 'dict' objects}
       21    0.015    0.001    0.015    0.001 surrogate.py:2131(<genexpr>)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        4    0.003    0.001    0.003    0.001 _function_base_impl.py:1413(diff)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         9040 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1722(__call__)
        1    0.002    0.002    0.033    0.033 surrogate.py:934(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
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
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         9017 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1722(__call__)
        1    0.002    0.002    0.034    0.034 surrogate.py:934(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.021    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.014    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
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
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         9017 function calls in 0.305 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.304    0.304 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.026    0.026    0.304    0.304 surrogate.py:1722(__call__)
        1    0.003    0.003    0.267    0.267 surrogate.py:934(__call__)
        1    0.008    0.008    0.240    0.240 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.101    0.101 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.100    0.100    0.101    0.101 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.073    0.073    0.073    0.073 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.051    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.051    0.025    0.051    0.026 spline_interp_Cwrapper.py:50(interpolate)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.014    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
        9    0.012    0.001    0.012    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1413(diff)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5592(append)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         9017 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1722(__call__)
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
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         9017 function calls in 0.092 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.092    0.092 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.092    0.092 surrogate.py:1722(__call__)
        1    0.002    0.002    0.088    0.088 surrogate.py:934(__call__)
        1    0.002    0.002    0.060    0.060 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.026    0.026 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.026    0.026    0.026    0.026 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
        1    0.020    0.020    0.020    0.020 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
       26    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         3540 function calls in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.022    0.022 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.022    0.022 surrogate.py:1722(__call__)
        1    0.000    0.000    0.020    0.020 surrogate.py:934(__call__)
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
       11    0.001    0.000    0.001    0.000 surrogate.py:2131(<genexpr>)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.000    0.000 surrogate.py:1635(_check_params)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         3536 function calls in 0.009 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.009    0.009 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.009    0.009 surrogate.py:1722(__call__)
        1    0.000    0.000    0.008    0.008 surrogate.py:934(__call__)
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
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       11    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         3540 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.037    0.037 surrogate.py:1722(__call__)
        1    0.000    0.000    0.032    0.032 surrogate.py:934(__call__)
        1    0.001    0.001    0.026    0.026 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.007    0.007 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.007    0.007    0.007    0.007 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.007    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.006    0.003    0.007    0.003 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.006    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.006    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2131(<genexpr>)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         3536 function calls in 0.011 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.011    0.011 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.011    0.011 surrogate.py:1722(__call__)
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
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       11    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         3523 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1722(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:934(__call__)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
        1    0.000    0.000    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:155(__call__)
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
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      332    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         3523 function calls in 0.163 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.163    0.163 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.013    0.013    0.163    0.163 surrogate.py:1722(__call__)
        1    0.000    0.000    0.143    0.143 surrogate.py:934(__call__)
        1    0.006    0.006    0.137    0.137 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.055    0.055    0.055    0.055 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.037    0.037 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.037    0.037    0.037    0.037 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.035    0.017 surrogate.py:91(_splinterp_Cwrapper)
        2    0.034    0.017    0.035    0.017 spline_interp_Cwrapper.py:50(interpolate)
        5    0.007    0.001    0.007    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.006    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.006    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.006    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1413(diff)
        2    0.001    0.001    0.001    0.001 surrogate.py:732(_search_omega)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1635(_check_params)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         3523 function calls in 0.009 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.008    0.008 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.008    0.008 surrogate.py:1722(__call__)
        1    0.000    0.000    0.008    0.008 surrogate.py:934(__call__)
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
        2    0.000    0.000    0.001    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      332    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         3523 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.038    0.038 surrogate.py:1722(__call__)
        1    0.000    0.000    0.035    0.035 surrogate.py:934(__call__)
        1    0.001    0.001    0.030    0.030 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.011    0.011    0.011    0.011 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.007    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      332    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:1635(_check_params)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         10972 function calls in 0.059 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.059    0.059 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.059    0.059 surrogate.py:1722(__call__)
        1    0.002    0.002    0.056    0.056 surrogate.py:934(__call__)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
        1    0.001    0.001    0.026    0.026 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.017    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.003 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       18    0.002    0.000    0.002    0.000 surrogate.py:2131(<genexpr>)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         10968 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.037    0.037 surrogate.py:1722(__call__)
        1    0.002    0.002    0.036    0.036 surrogate.py:934(__call__)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.017    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         10972 function calls in 0.095 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.095    0.095 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.095    0.095 surrogate.py:1722(__call__)
        1    0.003    0.003    0.080    0.080 surrogate.py:934(__call__)
        1    0.002    0.002    0.049    0.049 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.027    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.021    0.021 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.020    0.020    0.021    0.021 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
        1    0.016    0.016    0.016    0.016 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.012    0.012 {method 'update' of 'dict' objects}
       18    0.012    0.001    0.012    0.001 surrogate.py:2131(<genexpr>)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.009    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.004    0.009    0.005 spline_interp_Cwrapper.py:50(interpolate)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         10968 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1722(__call__)
        1    0.002    0.002    0.037    0.037 surrogate.py:934(__call__)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.017    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         10948 function calls in 0.072 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.072    0.072 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.072    0.072 surrogate.py:1722(__call__)
        1    0.003    0.003    0.071    0.071 surrogate.py:934(__call__)
       11    0.000    0.000    0.036    0.003 surrogate.py:425(_eval_sur)
       11    0.001    0.000    0.036    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.036    0.002 surrogate.py:292(__call__)
        1    0.004    0.004    0.030    0.030 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.000    0.000    0.025    0.000 nodeFunction.py:220(__call__)
      218    0.003    0.000    0.025    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.016    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.013    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.014    0.014 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.014    0.014    0.014    0.014 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       23    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.006    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.006    0.003 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.004    0.000 _py_warnings.py:254(filterwarnings)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
        6    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.001    0.000    0.001    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
     1090    0.001    0.000    0.001    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5592(append)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
       23    0.001    0.000    0.001    0.000 {built-in method numpy.array}
       30    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      237    0.000    0.000    0.000    0.000 saveH5Object.py:241(__iter__)
        1    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
       10    0.000    0.000    0.000    0.000 _internal.py:279(data_as)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         10948 function calls in 0.291 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.291    0.291 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.022    0.022    0.291    0.291 surrogate.py:1722(__call__)
        1    0.003    0.003    0.259    0.259 surrogate.py:934(__call__)
        1    0.008    0.008    0.228    0.228 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.091    0.091 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.091    0.091    0.091    0.091 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.069    0.069    0.069    0.069 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.052    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.052    0.026    0.052    0.026 spline_interp_Cwrapper.py:50(interpolate)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.027    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.017    0.000 nodeFunction.py:125(__call__)
        7    0.011    0.002    0.011    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1413(diff)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5592(append)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      654    0.000    0.000    0.000    0.000 {built-in method _warnings._acquire_lock}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         10948 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1722(__call__)
        1    0.002    0.002    0.035    0.035 surrogate.py:934(__call__)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      654    0.000    0.000    0.000    0.000 {built-in method _warnings._acquire_lock}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         10948 function calls in 0.090 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.090    0.090 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.090    0.090 surrogate.py:1722(__call__)
        1    0.002    0.002    0.087    0.087 surrogate.py:934(__call__)
        1    0.002    0.002    0.057    0.057 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.027    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.024    0.024 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.023    0.023    0.024    0.024 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.019    0.019    0.019    0.019 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
```
