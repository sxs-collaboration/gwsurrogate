# GWSurrogate Evaluation Timing

Generated: 2026-05-06T22:30:46.392410+00:00

Times below are seconds per model evaluation. Raw repeats and context are in the JSON output.

PNG timing table: `test/benchmark_surrogate_evaluations.png`

## Summary

### master

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0218847` s, median `0.0220419` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.022068` s, median `0.0223024` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0255053` s, median `0.0264275` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0254605` s, median `0.0255472` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0213691` s, median `0.0214589` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0214835` s, median `0.0215185` s
- `dt=0.5 M`, `f_low=0`: best `0.0173353` s, median `0.017497` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0202853` s, median `0.0205887` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0346106` s, median `0.0347597` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0338259` s, median `0.0346047` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0358262` s, median `0.0360468` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0347762` s, median `0.0348273` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0378106` s, median `0.0379131` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0358614` s, median `0.0360394` s
- `dt=0.5 M`, `f_low=0`: best `0.030932` s, median `0.0310369` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0345878` s, median `0.0346938` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.117736` s, median `0.118258` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0514421` s, median `0.0516358` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.198685` s, median `0.199581` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0556617` s, median `0.0559418` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0572327` s, median `0.0574294` s
- `dt=0.1 M`, `f_low=0.002`: best `0.682056` s, median `0.683251` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0498673` s, median `0.0499539` s
- `dt=0.5 M`, `f_low=0.002`: best `0.210122` s, median `0.211227` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0397392` s, median `0.0398186` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0150265` s, median `0.0152244` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0660799` s, median `0.066234` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0163227` s, median `0.0165253` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0171594` s, median `0.0172571` s
- `dt=0.1 M`, `f_low=0.002`: best `0.313974` s, median `0.314498` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0137626` s, median `0.0141895` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0718539` s, median `0.0721098` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.116` s, median `0.117077` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0558213` s, median `0.0561691` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.166396` s, median `0.166836` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0595289` s, median `0.0599438` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0612017` s, median `0.0615474` s
- `dt=0.1 M`, `f_low=0.002`: best `0.607826` s, median `0.610676` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0537612` s, median `0.0546616` s
- `dt=0.5 M`, `f_low=0.002`: best `0.193902` s, median `0.194396` s

### PR-71

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0222459` s, median `0.022291` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0221586` s, median `0.0223407` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0253661` s, median `0.0262228` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0253209` s, median `0.0254723` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.021506` s, median `0.0216283` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0215826` s, median `0.0216182` s
- `dt=0.5 M`, `f_low=0`: best `0.0175761` s, median `0.0177037` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0205055` s, median `0.0205797` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0341782` s, median `0.03433` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0337034` s, median `0.0337404` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0354891` s, median `0.0356801` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0342762` s, median `0.0344761` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0375368` s, median `0.0375804` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0352258` s, median `0.0355075` s
- `dt=0.5 M`, `f_low=0`: best `0.0304036` s, median `0.0304997` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0340833` s, median `0.0342529` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0813753` s, median `0.0815469` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.047713` s, median `0.0480789` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.121267` s, median `0.121392` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0495069` s, median `0.0499246` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0497991` s, median `0.0503416` s
- `dt=0.1 M`, `f_low=0.002`: best `0.322543` s, median `0.324559` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0466108` s, median `0.0486002` s
- `dt=0.5 M`, `f_low=0.002`: best `0.103082` s, median `0.103412` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0294852` s, median `0.0295146` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0150209` s, median `0.0152408` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0440149` s, median `0.0440984` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.015971` s, median `0.0161726` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0163701` s, median `0.0164797` s
- `dt=0.1 M`, `f_low=0.002`: best `0.18073` s, median `0.181524` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0143304` s, median `0.0144822` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0456289` s, median `0.0458423` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0781043` s, median `0.0789164` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0557964` s, median `0.0558098` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.10434` s, median `0.104513` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0564903` s, median `0.056694` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0572844` s, median `0.0577672` s
- `dt=0.1 M`, `f_low=0.002`: best `0.308729` s, median `0.309786` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0548184` s, median `0.0556771` s
- `dt=0.5 M`, `f_low=0.002`: best `0.107971` s, median `0.108016` s

## Context

### master

- Git branch: `master`
- Git commit: `95dd055d7917cb046a12e4226506a601c103db76`
- Git describe: `v1.1.8-19-g95dd055`
- Python: `3.14.4 (main, Apr  8 2026, 02:27:22) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1010-azure x86_64`
- CPU count: `4`
- Conda env: `unknown`

### PR-71

- Git branch: `unknown`
- Git commit: `8e3d6e8b585b5cfb5a6de5ad7c538943a4837e59`
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
         27100 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.007    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     2806    0.002    0.000    0.006    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
     2806    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1633    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         27101 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.007    0.007 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.007    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     2806    0.002    0.000    0.006    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
     2806    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.002    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:660(_integrate_backward)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     1633    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         27100 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1721(__call__)
        1    0.000    0.000    0.035    0.035 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:110(rotateWaveform)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.006    0.006    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.007    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     2806    0.002    0.000    0.006    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:769(__call__)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
     2806    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.004    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
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
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         27101 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1721(__call__)
        1    0.000    0.000    0.034    0.034 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:110(rotateWaveform)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.006    0.006    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.007    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     2806    0.002    0.000    0.006    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:769(__call__)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
     2806    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
     1633    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         20326 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 91 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:626(_integrate_forward)
      238    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        4    0.000    0.000    0.006    0.002 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.006    0.006 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.006    0.006    0.006    0.006 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      714    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2390    0.001    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
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
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         27707 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
      837    0.001    0.000    0.007    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
     2876    0.002    0.000    0.006    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_forward)
     2876    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
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
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         20326 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 91 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:626(_integrate_forward)
      238    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
      714    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2390    0.001    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2390    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.002    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     1252    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         27707 function calls in 0.030 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.030    0.030 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
      837    0.001    0.000    0.007    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
     2876    0.002    0.000    0.006    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
     2876    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.000    0.000    0.002    0.000 surrogate.py:105(_splinterp_Cwrapper_many)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1703    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
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
        1    0.000    0.000    0.048    0.048 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.048    0.048 surrogate.py:1721(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.021    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:626(_integrate_forward)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         43097 function calls in 0.047 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.021    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:626(_integrate_forward)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        4    0.000    0.000    0.003    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         43096 function calls in 0.050 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.050    0.050 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.050    0.050 surrogate.py:1721(__call__)
        1    0.000    0.000    0.049    0.049 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.049    0.049 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.021    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:626(_integrate_forward)
     5135    0.003    0.000    0.015    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:660(_integrate_backward)
        4    0.000    0.000    0.004    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         43097 function calls in 0.049 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.049    0.049 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.049    0.049 surrogate.py:1721(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.021    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:626(_integrate_forward)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
        4    0.000    0.000    0.003    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         35075 function calls in 0.049 seconds

   Ordered by: cumulative time
   List reduced from 94 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.049    0.049 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.049    0.049 surrogate.py:1721(__call__)
        1    0.000    0.000    0.049    0.049 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.049    0.049 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.021    0.021 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     4637    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
        4    0.000    0.000    0.011    0.003 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.010    0.010 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     2142    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        7    0.000    0.000    0.000    0.000 fromnumeric.py:2304(sum)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:864(normalize_spin)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         44452 function calls in 0.050 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.050    0.050 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.050    0.050 surrogate.py:1721(__call__)
        1    0.000    0.000    0.050    0.050 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.050    0.050 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.021    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.017    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5274    0.003    0.000    0.015    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.011    0.011 precessing_surrogate.py:660(_integrate_backward)
        1    0.001    0.001    0.011    0.011 precessing_surrogate.py:626(_integrate_forward)
     5274    0.009    0.000    0.009    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
        4    0.000    0.000    0.004    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:345(_get_t_from_omega)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:339(get_omega)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2744    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         35075 function calls in 0.042 seconds

   Ordered by: cumulative time
   List reduced from 94 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.042    0.042 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.042    0.042 surrogate.py:1721(__call__)
        1    0.000    0.000    0.042    0.042 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.042    0.042 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.021    0.021 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
     4637    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.004    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     2142    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:561(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:864(normalize_spin)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         44452 function calls in 0.048 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.048    0.048 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.048    0.048 surrogate.py:1721(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.021    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.017    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5274    0.003    0.000    0.015    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.011    0.011 precessing_surrogate.py:660(_integrate_backward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:626(_integrate_forward)
     5274    0.009    0.000    0.009    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        4    0.000    0.000    0.002    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:345(_get_t_from_omega)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:339(get_omega)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     2744    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         68518 function calls in 0.154 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.154    0.154 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.001    0.001    0.154    0.154 surrogate.py:1721(__call__)
        1    0.000    0.000    0.150    0.150 surrogate.py:938(__call__)
        1    0.059    0.059    0.082    0.082 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.068    0.006 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.068    0.006 surrogate.py:416(__call__)
       20    0.000    0.000    0.067    0.003 surrogate.py:291(__call__)
      158    0.000    0.000    0.060    0.000 nodeFunction.py:205(__call__)
      158    0.001    0.000    0.060    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.055    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.055    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.055    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.001    0.000    0.034    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.032    0.000 _gpr.py:373(predict)
      316    0.003    0.000    0.028    0.000 validation.py:725(check_array)
       12    0.000    0.000    0.021    0.002 surrogate.py:90(_splinterp_Cwrapper)
      158    0.000    0.000    0.021    0.000 _base.py:297(predict)
      158    0.001    0.000    0.021    0.000 _base.py:287(_decision_function)
       10    0.000    0.000    0.016    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.015    0.002    0.016    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      158    0.001    0.000    0.014    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.011    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.010    0.000 kernels.py:931(__call__)
      948    0.003    0.000    0.008    0.000 validation.py:371(_num_samples)
       24    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
        2    0.005    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
     6164    0.002    0.000    0.004    0.000 {built-in method builtins.isinstance}
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.003    0.000 _array_api.py:857(_asarray_with_order)
      474    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      632    0.001    0.000    0.002    0.000 _aliases.py:89(asarray)
      948    0.001    0.000    0.002    0.000 _array_api.py:331(get_namespace)
     1264    0.001    0.000    0.002    0.000 _config.py:35(get_config)
      474    0.001    0.000    0.002    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         68514 function calls in 0.086 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.087    0.087 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.087    0.087 surrogate.py:1721(__call__)
        1    0.000    0.000    0.086    0.086 surrogate.py:938(__call__)
       11    0.000    0.000    0.068    0.006 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.068    0.006 surrogate.py:416(__call__)
       20    0.000    0.000    0.068    0.003 surrogate.py:291(__call__)
      158    0.000    0.000    0.061    0.000 nodeFunction.py:205(__call__)
      158    0.001    0.000    0.061    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.056    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.055    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.055    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.001    0.000    0.034    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.033    0.000 _gpr.py:373(predict)
      316    0.004    0.000    0.028    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.021    0.000 _base.py:297(predict)
      158    0.001    0.000    0.021    0.000 _base.py:287(_decision_function)
        1    0.010    0.010    0.018    0.018 surrogate.py:741(_coorbital_to_inertial_frame)
      158    0.000    0.000    0.014    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.011    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.010    0.000 kernels.py:931(__call__)
      948    0.003    0.000    0.008    0.000 validation.py:371(_num_samples)
       24    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
       12    0.000    0.000    0.006    0.001 surrogate.py:90(_splinterp_Cwrapper)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
       10    0.000    0.000    0.005    0.000 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.004    0.000    0.005    0.000 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     6164    0.002    0.000    0.004    0.000 {built-in method builtins.isinstance}
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.003    0.000 _array_api.py:857(_asarray_with_order)
      474    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      948    0.001    0.000    0.002    0.000 _array_api.py:331(get_namespace)
      316    0.000    0.000    0.002    0.000 _py_warnings.py:294(simplefilter)
      474    0.001    0.000    0.002    0.000 base.py:603(__sklearn_tags__)
     1264    0.001    0.000    0.002    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         68518 function calls in 0.232 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.232    0.232 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.003    0.003    0.232    0.232 surrogate.py:1721(__call__)
        1    0.000    0.000    0.224    0.224 surrogate.py:938(__call__)
        1    0.110    0.110    0.156    0.156 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.067    0.006 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.067    0.006 surrogate.py:416(__call__)
       20    0.000    0.000    0.067    0.003 surrogate.py:291(__call__)
      158    0.000    0.000    0.060    0.000 nodeFunction.py:205(__call__)
      158    0.001    0.000    0.060    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.055    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.055    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.055    0.000 evaluate_fit.py:128(GPR_predict)
       12    0.000    0.000    0.043    0.004 surrogate.py:90(_splinterp_Cwrapper)
      316    0.002    0.000    0.034    0.000 validation.py:2793(validate_data)
       10    0.000    0.000    0.033    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.032    0.003    0.033    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      158    0.002    0.000    0.032    0.000 _gpr.py:373(predict)
      316    0.003    0.000    0.028    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.021    0.000 _base.py:297(predict)
      158    0.001    0.000    0.021    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.013    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.011    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.010    0.000 kernels.py:931(__call__)
        2    0.009    0.005    0.009    0.005 spline_interp_Cwrapper.py:50(interpolate)
      948    0.003    0.000    0.008    0.000 validation.py:371(_num_samples)
       24    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.002    0.000    0.004    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       21    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.003    0.000 _array_api.py:857(_asarray_with_order)
      474    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      632    0.001    0.000    0.002    0.000 _aliases.py:89(asarray)
      474    0.001    0.000    0.002    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         68514 function calls in 0.091 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.091    0.091 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.091    0.091 surrogate.py:1721(__call__)
        1    0.000    0.000    0.090    0.090 surrogate.py:938(__call__)
       11    0.000    0.000    0.068    0.006 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.068    0.006 surrogate.py:416(__call__)
       20    0.000    0.000    0.068    0.003 surrogate.py:291(__call__)
      158    0.000    0.000    0.061    0.000 nodeFunction.py:205(__call__)
      158    0.001    0.000    0.060    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.056    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.055    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.055    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.001    0.000    0.034    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.033    0.000 _gpr.py:373(predict)
      316    0.003    0.000    0.028    0.000 validation.py:725(check_array)
        1    0.013    0.013    0.022    0.022 surrogate.py:741(_coorbital_to_inertial_frame)
      158    0.000    0.000    0.021    0.000 _base.py:297(predict)
      158    0.001    0.000    0.021    0.000 _base.py:287(_decision_function)
      158    0.000    0.000    0.014    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.011    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.010    0.000 kernels.py:931(__call__)
      948    0.003    0.000    0.008    0.000 validation.py:371(_num_samples)
       12    0.000    0.000    0.007    0.001 surrogate.py:90(_splinterp_Cwrapper)
       24    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
       10    0.000    0.000    0.006    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.005    0.001    0.006    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     6164    0.002    0.000    0.004    0.000 {built-in method builtins.isinstance}
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.003    0.000 _array_api.py:857(_asarray_with_order)
      474    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      948    0.001    0.000    0.002    0.000 _array_api.py:331(get_namespace)
      316    0.000    0.000    0.002    0.000 fromnumeric.py:2304(sum)
      474    0.001    0.000    0.002    0.000 base.py:603(__sklearn_tags__)
      316    0.000    0.000    0.002    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         68491 function calls in 0.092 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.093    0.093 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.093    0.093 surrogate.py:1721(__call__)
        1    0.000    0.000    0.092    0.092 surrogate.py:938(__call__)
       11    0.000    0.000    0.068    0.006 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.068    0.006 surrogate.py:416(__call__)
       20    0.000    0.000    0.067    0.003 surrogate.py:291(__call__)
      158    0.000    0.000    0.061    0.000 nodeFunction.py:205(__call__)
      158    0.001    0.000    0.060    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.056    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.055    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.055    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.034    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.032    0.000 _gpr.py:373(predict)
      316    0.003    0.000    0.028    0.000 validation.py:725(check_array)
        1    0.015    0.015    0.024    0.024 surrogate.py:741(_coorbital_to_inertial_frame)
      158    0.000    0.000    0.021    0.000 _base.py:297(predict)
      158    0.001    0.000    0.021    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.014    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.011    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.010    0.000 kernels.py:931(__call__)
      948    0.003    0.000    0.008    0.000 validation.py:371(_num_samples)
       12    0.000    0.000    0.008    0.001 surrogate.py:90(_splinterp_Cwrapper)
       24    0.006    0.000    0.006    0.000 {method 'dot' of 'numpy.ndarray' objects}
       10    0.000    0.000    0.006    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.006    0.001    0.006    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.002    0.000    0.004    0.000 {built-in method builtins.isinstance}
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.003    0.000 _array_api.py:857(_asarray_with_order)
      474    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      948    0.001    0.000    0.002    0.000 _array_api.py:331(get_namespace)
      474    0.001    0.000    0.002    0.000 base.py:603(__sklearn_tags__)
      316    0.000    0.000    0.002    0.000 _py_warnings.py:294(simplefilter)
      316    0.000    0.000    0.002    0.000 fromnumeric.py:2304(sum)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         68491 function calls in 0.716 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.716    0.716 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.021    0.021    0.716    0.716 surrogate.py:1721(__call__)
        1    0.000    0.000    0.682    0.682 surrogate.py:938(__call__)
        1    0.440    0.440    0.614    0.614 surrogate.py:741(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.166    0.014 surrogate.py:90(_splinterp_Cwrapper)
       10    0.000    0.000    0.117    0.012 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.116    0.012    0.117    0.012 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       11    0.000    0.000    0.068    0.006 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.068    0.006 surrogate.py:416(__call__)
       20    0.000    0.000    0.068    0.003 surrogate.py:291(__call__)
      158    0.000    0.000    0.061    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.060    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.055    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.055    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.055    0.000 evaluate_fit.py:128(GPR_predict)
        2    0.047    0.024    0.048    0.024 spline_interp_Cwrapper.py:50(interpolate)
      316    0.001    0.000    0.034    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.032    0.000 _gpr.py:373(predict)
      316    0.003    0.000    0.028    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.021    0.000 _base.py:297(predict)
      158    0.001    0.000    0.021    0.000 _base.py:287(_decision_function)
      158    0.000    0.000    0.014    0.000 kernels.py:833(__call__)
        9    0.013    0.001    0.013    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      316    0.003    0.000    0.011    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.010    0.000 kernels.py:931(__call__)
      948    0.003    0.000    0.008    0.000 validation.py:371(_num_samples)
       24    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.002    0.000    0.004    0.000 {built-in method builtins.isinstance}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.003    0.000 _array_api.py:857(_asarray_with_order)
      474    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      474    0.001    0.000    0.002    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         68491 function calls in 0.094 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.095    0.095 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.095    0.095 surrogate.py:1721(__call__)
        1    0.000    0.000    0.095    0.095 surrogate.py:938(__call__)
       11    0.000    0.000    0.077    0.007 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.077    0.007 surrogate.py:416(__call__)
       20    0.000    0.000    0.077    0.004 surrogate.py:291(__call__)
      158    0.000    0.000    0.062    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.062    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.057    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.056    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.056    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.035    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.033    0.000 _gpr.py:373(predict)
      316    0.004    0.000    0.028    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.021    0.000 _base.py:297(predict)
      158    0.001    0.000    0.021    0.000 _base.py:287(_decision_function)
        1    0.010    0.010    0.017    0.017 surrogate.py:741(_coorbital_to_inertial_frame)
       24    0.015    0.001    0.015    0.001 {method 'dot' of 'numpy.ndarray' objects}
      158    0.001    0.000    0.014    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.011    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.010    0.000 kernels.py:931(__call__)
      948    0.003    0.000    0.008    0.000 validation.py:371(_num_samples)
       12    0.000    0.000    0.006    0.001 surrogate.py:90(_splinterp_Cwrapper)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
       10    0.000    0.000    0.005    0.000 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.004    0.000    0.005    0.000 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     6164    0.002    0.000    0.004    0.000 {built-in method builtins.isinstance}
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.003    0.000 _array_api.py:857(_asarray_with_order)
      474    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      316    0.001    0.000    0.002    0.000 fromnumeric.py:2304(sum)
      474    0.001    0.000    0.002    0.000 base.py:603(__sklearn_tags__)
      948    0.001    0.000    0.002    0.000 _array_api.py:331(get_namespace)
      316    0.000    0.000    0.002    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         68491 function calls in 0.244 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.244    0.244 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.004    0.004    0.244    0.244 surrogate.py:1721(__call__)
        1    0.000    0.000    0.238    0.238 surrogate.py:938(__call__)
        1    0.124    0.124    0.171    0.171 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.067    0.006 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.067    0.006 surrogate.py:416(__call__)
       20    0.000    0.000    0.067    0.003 surrogate.py:291(__call__)
      158    0.000    0.000    0.060    0.000 nodeFunction.py:205(__call__)
      158    0.001    0.000    0.060    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.055    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.055    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.055    0.000 evaluate_fit.py:128(GPR_predict)
       12    0.000    0.000    0.043    0.004 surrogate.py:90(_splinterp_Cwrapper)
      316    0.001    0.000    0.034    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.032    0.000 _gpr.py:373(predict)
       10    0.000    0.000    0.031    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.029    0.003    0.031    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      316    0.003    0.000    0.028    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.021    0.000 _base.py:297(predict)
      158    0.001    0.000    0.021    0.000 _base.py:287(_decision_function)
      158    0.000    0.000    0.014    0.000 kernels.py:833(__call__)
        2    0.011    0.006    0.012    0.006 spline_interp_Cwrapper.py:50(interpolate)
      316    0.003    0.000    0.011    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.010    0.000 kernels.py:931(__call__)
      948    0.003    0.000    0.008    0.000 validation.py:371(_num_samples)
       24    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.002    0.000    0.004    0.000 {built-in method builtins.isinstance}
      158    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.003    0.000 _array_api.py:857(_asarray_with_order)
      474    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      474    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      632    0.001    0.000    0.002    0.000 _aliases.py:89(asarray)
      948    0.001    0.000    0.002    0.000 _array_api.py:331(get_namespace)
      474    0.001    0.000    0.002    0.000 base.py:603(__sklearn_tags__)
      316    0.000    0.000    0.002    0.000 fromnumeric.py:2304(sum)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         21866 function calls in 0.054 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.055    0.055 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.001    0.001    0.055    0.055 surrogate.py:1721(__call__)
        1    0.000    0.000    0.052    0.052 surrogate.py:938(__call__)
        1    0.019    0.019    0.031    0.031 surrogate.py:741(_coorbital_to_inertial_frame)
        5    0.000    0.000    0.022    0.004 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.021    0.004 surrogate.py:416(__call__)
       10    0.000    0.000    0.021    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.020    0.000 nodeFunction.py:205(__call__)
       50    0.001    0.000    0.020    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.018    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.018    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.018    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.001    0.000    0.011    0.000 validation.py:2793(validate_data)
       50    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
        6    0.000    0.000    0.010    0.002 surrogate.py:90(_splinterp_Cwrapper)
      100    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.007    0.000 _base.py:297(predict)
       50    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
        4    0.000    0.000    0.006    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.006    0.001    0.006    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
        2    0.004    0.002    0.004    0.002 spline_interp_Cwrapper.py:50(interpolate)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       14    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
       11    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      100    0.000    0.000    0.001    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         21862 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 surrogate.py:938(__call__)
        5    0.000    0.000    0.017    0.003 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.017    0.003 surrogate.py:416(__call__)
       10    0.000    0.000    0.017    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.014    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.014    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.014    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.009    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.008    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.007    0.000 validation.py:725(check_array)
        1    0.002    0.002    0.007    0.007 surrogate.py:741(_coorbital_to_inertial_frame)
       50    0.000    0.000    0.005    0.000 _base.py:297(predict)
       50    0.000    0.000    0.005    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
        6    0.000    0.000    0.003    0.001 surrogate.py:90(_splinterp_Cwrapper)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
        4    0.000    0.000    0.002    0.000 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.002    0.000    0.002    0.000 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
     1952    0.000    0.000    0.001    0.000 {built-in method builtins.isinstance}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      150    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
     2950    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         21866 function calls in 0.076 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.076    0.076 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.001    0.001    0.076    0.076 surrogate.py:1721(__call__)
        1    0.000    0.000    0.072    0.072 surrogate.py:938(__call__)
        1    0.036    0.036    0.055    0.055 surrogate.py:741(_coorbital_to_inertial_frame)
        5    0.000    0.000    0.017    0.003 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.017    0.003 surrogate.py:416(__call__)
       10    0.000    0.000    0.017    0.002 surrogate.py:291(__call__)
        6    0.000    0.000    0.017    0.003 surrogate.py:90(_splinterp_Cwrapper)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.014    0.000 evaluate_fit.py:128(GPR_predict)
        4    0.000    0.000    0.009    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.009    0.002    0.009    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      100    0.000    0.000    0.009    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.007    0.000 validation.py:725(check_array)
        2    0.007    0.003    0.007    0.003 spline_interp_Cwrapper.py:50(interpolate)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         21862 function calls in 0.026 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.026    0.026 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.026    0.026 surrogate.py:1721(__call__)
        1    0.000    0.000    0.026    0.026 surrogate.py:938(__call__)
        5    0.000    0.000    0.017    0.003 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.017    0.003 surrogate.py:416(__call__)
       10    0.000    0.000    0.017    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.014    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.014    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.009    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
        1    0.003    0.003    0.008    0.008 surrogate.py:741(_coorbital_to_inertial_frame)
      100    0.001    0.000    0.007    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.005    0.000 _base.py:297(predict)
       50    0.000    0.000    0.005    0.000 _base.py:287(_decision_function)
        6    0.000    0.000    0.004    0.001 surrogate.py:90(_splinterp_Cwrapper)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.002    0.000    0.002    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
     1952    0.000    0.000    0.001    0.000 {built-in method builtins.isinstance}
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       50    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
     2950    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         21849 function calls in 0.027 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.027    0.027 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.027    0.027 surrogate.py:1721(__call__)
        1    0.000    0.000    0.027    0.027 surrogate.py:938(__call__)
        5    0.000    0.000    0.017    0.003 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.017    0.003 surrogate.py:416(__call__)
       10    0.000    0.000    0.017    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.014    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.004    0.004    0.009    0.009 surrogate.py:741(_coorbital_to_inertial_frame)
      100    0.000    0.000    0.009    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.007    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.005    0.000 _base.py:287(_decision_function)
        6    0.000    0.000    0.004    0.001 surrogate.py:90(_splinterp_Cwrapper)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.002    0.001    0.002    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
     1952    0.000    0.000    0.001    0.000 {built-in method builtins.isinstance}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      150    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
     2950    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         21849 function calls in 0.322 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.323    0.323 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.012    0.012    0.323    0.323 surrogate.py:1721(__call__)
        1    0.000    0.000    0.303    0.303 surrogate.py:938(__call__)
        1    0.198    0.198    0.286    0.286 surrogate.py:741(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.083    0.014 surrogate.py:90(_splinterp_Cwrapper)
        4    0.000    0.000    0.046    0.011 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.045    0.011    0.046    0.011 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.037    0.018    0.037    0.019 spline_interp_Cwrapper.py:50(interpolate)
        5    0.000    0.000    0.017    0.003 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.017    0.003 surrogate.py:416(__call__)
       10    0.000    0.000    0.017    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.009    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.007    0.000 validation.py:725(check_array)
        5    0.007    0.001    0.007    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1402(diff)
        2    0.002    0.001    0.002    0.001 surrogate.py:731(_search_omega)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
     1952    0.000    0.000    0.001    0.000 {built-in method builtins.isinstance}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         21849 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 surrogate.py:938(__call__)
        5    0.000    0.000    0.017    0.003 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.017    0.003 surrogate.py:416(__call__)
       10    0.000    0.000    0.017    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.014    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.014    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.014    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.009    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.007    0.000 validation.py:725(check_array)
        1    0.002    0.002    0.007    0.007 surrogate.py:741(_coorbital_to_inertial_frame)
       50    0.000    0.000    0.005    0.000 _base.py:297(predict)
       50    0.000    0.000    0.005    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
        6    0.000    0.000    0.003    0.001 surrogate.py:90(_splinterp_Cwrapper)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
        4    0.000    0.000    0.002    0.000 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.001    0.000    0.002    0.000 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      300    0.000    0.000    0.001    0.000 _array_api.py:331(get_namespace)
       50    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
     2950    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         21849 function calls in 0.082 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.082    0.082 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.001    0.001    0.082    0.082 surrogate.py:1721(__call__)
        1    0.000    0.000    0.080    0.080 surrogate.py:938(__call__)
        1    0.041    0.041    0.062    0.062 surrogate.py:741(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.019    0.003 surrogate.py:90(_splinterp_Cwrapper)
        5    0.000    0.000    0.017    0.003 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.017    0.003 surrogate.py:416(__call__)
       10    0.000    0.000    0.017    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
        4    0.000    0.000    0.011    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.010    0.003    0.011    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      100    0.000    0.000    0.009    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
        2    0.008    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
      100    0.001    0.000    0.007    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.005    0.000 _base.py:297(predict)
       50    0.000    0.000    0.005    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
     1952    0.000    0.000    0.001    0.000 {built-in method builtins.isinstance}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      150    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         81324 function calls in 0.159 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.160    0.160 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.001    0.001    0.160    0.160 surrogate.py:1721(__call__)
        1    0.000    0.000    0.156    0.156 surrogate.py:938(__call__)
       10    0.000    0.000    0.081    0.008 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.081    0.008 surrogate.py:416(__call__)
       17    0.000    0.000    0.081    0.005 surrogate.py:291(__call__)
        1    0.052    0.052    0.074    0.074 surrogate.py:741(_coorbital_to_inertial_frame)
      188    0.000    0.000    0.073    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.073    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.067    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.067    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.066    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.041    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
      376    0.004    0.000    0.033    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.025    0.000 _base.py:297(predict)
      188    0.001    0.000    0.025    0.000 _base.py:287(_decision_function)
       11    0.000    0.000    0.020    0.002 surrogate.py:90(_splinterp_Cwrapper)
      188    0.001    0.000    0.017    0.000 kernels.py:833(__call__)
        9    0.000    0.000    0.015    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.014    0.002    0.015    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      376    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.013    0.000 kernels.py:931(__call__)
     1128    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
       21    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
        2    0.004    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
      188    0.001    0.000    0.004    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      564    0.001    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      564    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
    11092    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      752    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     1128    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      564    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      376    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         81320 function calls in 0.097 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.098    0.098 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.098    0.098 surrogate.py:1721(__call__)
        1    0.000    0.000    0.097    0.097 surrogate.py:938(__call__)
       10    0.000    0.000    0.080    0.008 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.080    0.008 surrogate.py:416(__call__)
       17    0.000    0.000    0.080    0.005 surrogate.py:291(__call__)
      188    0.000    0.000    0.072    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.072    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.066    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.066    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.066    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.040    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
      376    0.004    0.000    0.033    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.025    0.000 _base.py:297(predict)
      188    0.001    0.000    0.025    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.017    0.000 kernels.py:833(__call__)
        1    0.009    0.009    0.016    0.016 surrogate.py:741(_coorbital_to_inertial_frame)
      376    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.013    0.000 kernels.py:931(__call__)
     1128    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
       21    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
       11    0.000    0.000    0.006    0.001 surrogate.py:90(_splinterp_Cwrapper)
     7334    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
        9    0.000    0.000    0.004    0.000 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.004    0.000    0.004    0.000 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      564    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      564    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
    11092    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      752    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      564    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     1128    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      376    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      376    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         81324 function calls in 0.207 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.208    0.208 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.001    0.001    0.208    0.208 surrogate.py:1721(__call__)
        1    0.000    0.000    0.202    0.202 surrogate.py:938(__call__)
        1    0.087    0.087    0.121    0.121 surrogate.py:741(_coorbital_to_inertial_frame)
       10    0.000    0.000    0.080    0.008 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.080    0.008 surrogate.py:416(__call__)
       17    0.000    0.000    0.080    0.005 surrogate.py:291(__call__)
      188    0.000    0.000    0.071    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.071    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.066    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.065    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.065    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.040    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
       11    0.000    0.000    0.033    0.003 surrogate.py:90(_splinterp_Cwrapper)
      376    0.004    0.000    0.033    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.025    0.000 _base.py:297(predict)
      188    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
        9    0.000    0.000    0.024    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.024    0.003    0.024    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      188    0.001    0.000    0.017    0.000 kernels.py:833(__call__)
      188    0.001    0.000    0.013    0.000 kernels.py:931(__call__)
      376    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
     1128    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
        2    0.008    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
       21    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
     7334    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      188    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      564    0.001    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       18    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      564    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
    11092    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      752    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     1128    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         81320 function calls in 0.102 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.102    0.102 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.102    0.102 surrogate.py:1721(__call__)
        1    0.000    0.000    0.101    0.101 surrogate.py:938(__call__)
       10    0.000    0.000    0.081    0.008 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.081    0.008 surrogate.py:416(__call__)
       17    0.000    0.000    0.081    0.005 surrogate.py:291(__call__)
      188    0.000    0.000    0.073    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.072    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.067    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.066    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.066    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.041    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
      376    0.004    0.000    0.033    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.025    0.000 _base.py:297(predict)
      188    0.001    0.000    0.025    0.000 _base.py:287(_decision_function)
        1    0.012    0.012    0.020    0.020 surrogate.py:741(_coorbital_to_inertial_frame)
      188    0.001    0.000    0.017    0.000 kernels.py:833(__call__)
      188    0.001    0.000    0.013    0.000 kernels.py:931(__call__)
      376    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
     1128    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
       21    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
       11    0.000    0.000    0.007    0.001 surrogate.py:90(_splinterp_Cwrapper)
        9    0.000    0.000    0.005    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.005    0.001    0.005    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     7334    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      188    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      564    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
    11092    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      752    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     1128    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      564    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      376    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      376    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         81300 function calls in 0.104 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.104    0.104 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.104    0.104 surrogate.py:1721(__call__)
        1    0.000    0.000    0.104    0.104 surrogate.py:938(__call__)
       10    0.000    0.000    0.081    0.008 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.081    0.008 surrogate.py:416(__call__)
       17    0.000    0.000    0.081    0.005 surrogate.py:291(__call__)
      188    0.000    0.000    0.073    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.073    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.067    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.067    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.066    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.041    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
      376    0.004    0.000    0.033    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.025    0.000 _base.py:297(predict)
      188    0.001    0.000    0.025    0.000 _base.py:287(_decision_function)
        1    0.013    0.013    0.022    0.022 surrogate.py:741(_coorbital_to_inertial_frame)
      188    0.001    0.000    0.017    0.000 kernels.py:833(__call__)
      188    0.001    0.000    0.013    0.000 kernels.py:931(__call__)
      376    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
     1128    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
      188    0.004    0.000    0.008    0.000 kernels.py:1525(__call__)
       21    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
       11    0.000    0.000    0.007    0.001 surrogate.py:90(_splinterp_Cwrapper)
        9    0.000    0.000    0.006    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.005    0.001    0.005    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     7334    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      188    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      564    0.001    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
    11092    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      752    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     1128    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      564    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      376    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      376    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         81300 function calls in 0.646 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.647    0.647 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.017    0.017    0.647    0.647 surrogate.py:1721(__call__)
        1    0.000    0.000    0.620    0.620 surrogate.py:938(__call__)
        1    0.383    0.383    0.539    0.539 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.148    0.013 surrogate.py:90(_splinterp_Cwrapper)
        9    0.000    0.000    0.102    0.011 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.101    0.011    0.102    0.011 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       10    0.000    0.000    0.081    0.008 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.080    0.008 surrogate.py:416(__call__)
       17    0.000    0.000    0.080    0.005 surrogate.py:291(__call__)
      188    0.000    0.000    0.072    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.072    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.066    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.066    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.066    0.000 evaluate_fit.py:128(GPR_predict)
        2    0.045    0.023    0.046    0.023 spline_interp_Cwrapper.py:50(interpolate)
      376    0.002    0.000    0.040    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
      376    0.004    0.000    0.033    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.025    0.000 _base.py:297(predict)
      188    0.001    0.000    0.025    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.017    0.000 kernels.py:833(__call__)
      188    0.001    0.000    0.013    0.000 kernels.py:931(__call__)
      376    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
        7    0.010    0.001    0.010    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
     1128    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
       21    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      188    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
      376    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      564    0.001    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      188    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      564    0.001    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
    11092    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      752    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     1128    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         81300 function calls in 0.096 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.097    0.097 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.097    0.097 surrogate.py:1721(__call__)
        1    0.000    0.000    0.097    0.097 surrogate.py:938(__call__)
       10    0.000    0.000    0.081    0.008 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.081    0.008 surrogate.py:416(__call__)
       17    0.000    0.000    0.081    0.005 surrogate.py:291(__call__)
      188    0.000    0.000    0.073    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.072    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.067    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.067    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.066    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.041    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
      376    0.004    0.000    0.033    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.025    0.000 _base.py:297(predict)
      188    0.001    0.000    0.025    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.017    0.000 kernels.py:833(__call__)
        1    0.009    0.009    0.015    0.015 surrogate.py:741(_coorbital_to_inertial_frame)
      376    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.013    0.000 kernels.py:931(__call__)
     1128    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
       21    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
       11    0.000    0.000    0.006    0.001 surrogate.py:90(_splinterp_Cwrapper)
     7334    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
        9    0.000    0.000    0.004    0.000 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.004    0.000    0.004    0.000 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      376    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
    11092    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      752    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     1128    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
     1504    0.001    0.000    0.003    0.000 _config.py:35(get_config)
      564    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      376    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         81300 function calls in 0.234 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.235    0.235 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.006    0.006    0.235    0.235 surrogate.py:1721(__call__)
        1    0.000    0.000    0.228    0.228 surrogate.py:938(__call__)
        1    0.108    0.108    0.146    0.146 surrogate.py:741(_coorbital_to_inertial_frame)
       10    0.000    0.000    0.081    0.008 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.080    0.008 surrogate.py:416(__call__)
       17    0.000    0.000    0.080    0.005 surrogate.py:291(__call__)
      188    0.000    0.000    0.072    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.072    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.066    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.066    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.066    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.040    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.039    0.000 _gpr.py:373(predict)
       11    0.000    0.000    0.037    0.003 surrogate.py:90(_splinterp_Cwrapper)
      376    0.004    0.000    0.033    0.000 validation.py:725(check_array)
        9    0.000    0.000    0.027    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.026    0.003    0.026    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      188    0.000    0.000    0.025    0.000 _base.py:297(predict)
      188    0.001    0.000    0.025    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.017    0.000 kernels.py:833(__call__)
      188    0.001    0.000    0.013    0.000 kernels.py:931(__call__)
      376    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
        2    0.009    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
     1128    0.004    0.000    0.010    0.000 validation.py:371(_num_samples)
       21    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      188    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      564    0.001    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      188    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      564    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
    11092    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      752    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      376    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      564    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     1128    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
```

#### PR-71

##### NRSur7dq4 / mks_dt_0.000244140625_flow_0

```text
         27100 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.007    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     2806    0.002    0.000    0.006    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     2806    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1633    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         27101 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.007    0.007 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.007    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     2806    0.002    0.000    0.006    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     2806    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:660(_integrate_backward)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     1633    0.001    0.000    0.001    0.000 {built-in method numpy.array}
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
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         27100 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1721(__call__)
        1    0.000    0.000    0.035    0.035 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:110(rotateWaveform)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.006    0.006    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.007    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     2806    0.002    0.000    0.006    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:769(__call__)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     2806    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
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
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         27101 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1721(__call__)
        1    0.000    0.000    0.034    0.034 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:110(rotateWaveform)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.006    0.006    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.007    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     2806    0.002    0.000    0.006    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:769(__call__)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     2806    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
     1633    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         20326 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 91 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:626(_integrate_forward)
      238    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        4    0.000    0.000    0.006    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.006    0.006 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.006    0.006    0.006    0.006 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      714    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2390    0.001    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
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
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         27707 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
      837    0.001    0.000    0.007    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     2876    0.002    0.000    0.006    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_forward)
     2876    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
     1703    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         20326 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 91 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:626(_integrate_forward)
      238    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
      714    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2390    0.001    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
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
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         27707 function calls in 0.030 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.030    0.030 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.030    0.030 surrogate.py:1721(__call__)
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
      837    0.001    0.000    0.007    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
     2876    0.002    0.000    0.006    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     2876    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
     1703    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         43096 function calls in 0.047 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
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
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         43097 function calls in 0.046 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.046    0.046 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.046    0.046 surrogate.py:1721(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         43096 function calls in 0.050 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.050    0.050 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.050    0.050 surrogate.py:1721(__call__)
        1    0.000    0.000    0.050    0.050 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.050    0.050 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        4    0.000    0.000    0.005    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:660(_integrate_backward)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
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
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         35075 function calls in 0.048 seconds

   Ordered by: cumulative time
   List reduced from 94 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.048    0.048 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.048    0.048 surrogate.py:1721(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.020    0.020 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     4637    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
        4    0.000    0.000    0.011    0.003 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
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
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         44452 function calls in 0.049 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.049    0.049 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.049    0.049 surrogate.py:1721(__call__)
        1    0.000    0.000    0.049    0.049 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.049    0.049 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:392(__call__)
      546    0.002    0.000    0.021    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5274    0.003    0.000    0.015    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.011    0.011 precessing_surrogate.py:660(_integrate_backward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:626(_integrate_forward)
     5274    0.009    0.000    0.009    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:345(_get_t_from_omega)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:339(get_omega)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2744    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         35075 function calls in 0.041 seconds

   Ordered by: cumulative time
   List reduced from 94 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.041    0.041 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.041    0.041 surrogate.py:1721(__call__)
        1    0.000    0.000    0.041    0.041 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.041    0.041 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.021    0.021 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.020    0.020 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     4637    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
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
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         44452 function calls in 0.047 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.021    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5274    0.003    0.000    0.015    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:660(_integrate_backward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:626(_integrate_forward)
     5274    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:345(_get_t_from_omega)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:339(get_omega)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     2744    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         76481 function calls in 0.119 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.119    0.119 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.001    0.001    0.119    0.119 surrogate.py:1721(__call__)
        1    0.003    0.003    0.113    0.113 surrogate.py:934(__call__)
       12    0.000    0.000    0.079    0.007 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.078    0.007 surrogate.py:417(__call__)
       22    0.000    0.000    0.078    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.069    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.069    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.063    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.063    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.063    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.039    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.037    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.032    0.000 validation.py:725(check_array)
        1    0.001    0.001    0.031    0.031 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
        1    0.000    0.000    0.014    0.014 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.013    0.013    0.014    0.014 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
       26    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6905    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.005    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       21    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
      531    0.001    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         76477 function calls in 0.087 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.088    0.088 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.088    0.088 surrogate.py:1721(__call__)
        1    0.002    0.002    0.087    0.087 surrogate.py:934(__call__)
       12    0.000    0.000    0.077    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.077    0.006 surrogate.py:417(__call__)
       22    0.000    0.000    0.077    0.003 surrogate.py:292(__call__)
      177    0.000    0.000    0.069    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.069    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.064    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.063    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.063    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.039    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.037    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.032    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
     6905    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      531    0.001    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      354    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      354    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
     1416    0.001    0.000    0.003    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         76481 function calls in 0.160 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.160    0.160 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.002    0.002    0.160    0.160 surrogate.py:1721(__call__)
        1    0.003    0.003    0.140    0.140 surrogate.py:934(__call__)
       12    0.000    0.000    0.079    0.007 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.079    0.007 surrogate.py:417(__call__)
       22    0.000    0.000    0.079    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.070    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.069    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.064    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.063    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.063    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.003    0.003    0.057    0.057 surrogate.py:742(_coorbital_to_inertial_frame)
      354    0.002    0.000    0.039    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.037    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.032    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.023    0.023    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.001    0.000    0.015    0.000 kernels.py:833(__call__)
        1    0.000    0.000    0.015    0.015 {method 'update' of 'dict' objects}
       21    0.015    0.001    0.015    0.001 surrogate.py:2126(<genexpr>)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
        2    0.000    0.000    0.009    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.009    0.005 spline_interp_Cwrapper.py:50(interpolate)
       26    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6905    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      531    0.001    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
        4    0.003    0.001    0.003    0.001 _function_base_impl.py:1402(diff)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         76477 function calls in 0.089 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.089    0.089 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.089    0.089 surrogate.py:1721(__call__)
        1    0.002    0.002    0.088    0.088 surrogate.py:934(__call__)
       12    0.000    0.000    0.077    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.077    0.006 surrogate.py:417(__call__)
       22    0.000    0.000    0.077    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.069    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.069    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.063    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.063    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.063    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.039    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.037    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.032    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.015    0.000 kernels.py:833(__call__)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6905    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      531    0.001    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      354    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      354    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
     1416    0.001    0.000    0.003    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         76454 function calls in 0.089 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.090    0.090 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.090    0.090 surrogate.py:1721(__call__)
        1    0.002    0.002    0.089    0.089 surrogate.py:934(__call__)
       12    0.000    0.000    0.077    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.077    0.006 surrogate.py:417(__call__)
       22    0.000    0.000    0.077    0.003 surrogate.py:292(__call__)
      177    0.000    0.000    0.069    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.069    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.063    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.063    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.063    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.039    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.037    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.032    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6905    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      531    0.001    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      354    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      354    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         76454 function calls in 0.361 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.362    0.362 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.025    0.025    0.362    0.362 surrogate.py:1721(__call__)
        1    0.003    0.003    0.323    0.323 surrogate.py:934(__call__)
        1    0.009    0.009    0.241    0.241 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.101    0.101 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.101    0.101    0.101    0.101 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.078    0.007 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.078    0.007 surrogate.py:417(__call__)
       22    0.000    0.000    0.078    0.004 surrogate.py:292(__call__)
        1    0.075    0.075    0.075    0.075 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.069    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.069    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.063    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.063    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.063    0.000 evaluate_fit.py:128(GPR_predict)
        2    0.000    0.000    0.048    0.024 surrogate.py:91(_splinterp_Cwrapper)
        2    0.048    0.024    0.048    0.024 spline_interp_Cwrapper.py:50(interpolate)
      354    0.002    0.000    0.039    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.037    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.032    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.015    0.000 kernels.py:833(__call__)
        9    0.013    0.001    0.013    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      354    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6905    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      531    0.001    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         76454 function calls in 0.086 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.087    0.087 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.087    0.087 surrogate.py:1721(__call__)
        1    0.002    0.002    0.086    0.086 surrogate.py:934(__call__)
       12    0.000    0.000    0.078    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.078    0.006 surrogate.py:417(__call__)
       22    0.000    0.000    0.077    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.069    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.069    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.063    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.063    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.063    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.039    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.037    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.032    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
     6905    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      531    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      354    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      354    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
     1416    0.001    0.000    0.003    0.000 _config.py:35(get_config)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         76454 function calls in 0.143 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.143    0.143 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.002    0.002    0.143    0.143 surrogate.py:1721(__call__)
        1    0.002    0.002    0.139    0.139 surrogate.py:934(__call__)
       12    0.000    0.000    0.078    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.078    0.006 surrogate.py:417(__call__)
       22    0.000    0.000    0.078    0.004 surrogate.py:292(__call__)
      177    0.000    0.000    0.069    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.069    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.063    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.063    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.063    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.002    0.002    0.058    0.058 surrogate.py:742(_coorbital_to_inertial_frame)
      354    0.002    0.000    0.039    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.037    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.032    0.000 validation.py:725(check_array)
        1    0.000    0.000    0.025    0.025 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.024    0.024    0.025    0.025 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
        1    0.020    0.020    0.020    0.020 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.001    0.000    0.015    0.000 kernels.py:833(__call__)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.011    0.000 kernels.py:931(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6905    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      531    0.001    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      354    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         28686 function calls in 0.042 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.042    0.042 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.001    0.001    0.042    0.042 surrogate.py:1721(__call__)
        1    0.000    0.000    0.039    0.039 surrogate.py:934(__call__)
        6    0.000    0.000    0.023    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.022    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.021    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.021    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.019    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.001    0.001    0.016    0.016 surrogate.py:742(_coorbital_to_inertial_frame)
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.010    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
        1    0.005    0.005    0.005    0.005 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
        1    0.000    0.000    0.005    0.005 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.004    0.004    0.005    0.005 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.004    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.004    0.002 spline_interp_Cwrapper.py:50(interpolate)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.004    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       11    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         28682 function calls in 0.027 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.027    0.027 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.027    0.027 surrogate.py:1721(__call__)
        1    0.000    0.000    0.027    0.027 surrogate.py:934(__call__)
        6    0.000    0.000    0.022    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.022    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.021    0.000 nodeFunction.py:205(__call__)
       66    0.000    0.000    0.020    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.019    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.010    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      132    0.000    0.000    0.001    0.000 _py_warnings.py:294(simplefilter)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
     3894    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
      396    0.000    0.000    0.001    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         28686 function calls in 0.056 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.056    0.056 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.001    0.001    0.056    0.056 surrogate.py:1721(__call__)
        1    0.000    0.000    0.052    0.052 surrogate.py:934(__call__)
        1    0.002    0.002    0.029    0.029 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.022    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.022    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.021    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.021    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.019    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
        1    0.010    0.010    0.010    0.010 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      132    0.001    0.000    0.010    0.000 validation.py:725(check_array)
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
        2    0.000    0.000    0.007    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.007    0.003    0.007    0.004 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         28682 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 surrogate.py:934(__call__)
        6    0.000    0.000    0.022    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.022    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.021    0.000 nodeFunction.py:205(__call__)
       66    0.000    0.000    0.020    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.019    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      198    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
     3894    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         28669 function calls in 0.029 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.029    0.029 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.000    0.000    0.029    0.029 surrogate.py:934(__call__)
        6    0.000    0.000    0.022    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.022    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.021    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.021    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.019    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.010    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.002    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
     3894    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
      198    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         28669 function calls in 0.192 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.192    0.192 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.012    0.012    0.192    0.192 surrogate.py:1721(__call__)
        1    0.000    0.000    0.173    0.173 surrogate.py:934(__call__)
        1    0.006    0.006    0.150    0.150 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.061    0.061    0.061    0.061 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.041    0.041 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.041    0.041    0.041    0.041 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.037    0.018 surrogate.py:91(_splinterp_Cwrapper)
        2    0.036    0.018    0.037    0.018 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.022    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.022    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.021    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.021    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.019    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.010    0.000 validation.py:725(check_array)
        5    0.007    0.001    0.007    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.004    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1402(diff)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         28669 function calls in 0.027 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.027    0.027 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.027    0.027 surrogate.py:1721(__call__)
        1    0.000    0.000    0.027    0.027 surrogate.py:934(__call__)
        6    0.000    0.000    0.022    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.022    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.021    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.021    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.019    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.010    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.004    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
      396    0.000    0.000    0.001    0.000 _array_api.py:331(get_namespace)
     3894    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         28669 function calls in 0.058 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.058    0.058 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.001    0.001    0.058    0.058 surrogate.py:1721(__call__)
        1    0.000    0.000    0.056    0.056 surrogate.py:934(__call__)
        1    0.001    0.001    0.033    0.033 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.023    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.022    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.021    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.021    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.019    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.012    0.012    0.012    0.012 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.010    0.000 validation.py:725(check_array)
        1    0.000    0.000    0.009    0.009 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.009    0.009 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.008    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.004    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         94030 function calls in 0.127 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.128    0.128 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.001    0.001    0.128    0.128 surrogate.py:1721(__call__)
        1    0.002    0.002    0.124    0.124 surrogate.py:934(__call__)
       11    0.000    0.000    0.095    0.009 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.095    0.009 surrogate.py:417(__call__)
       19    0.000    0.000    0.095    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.085    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.078    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.030    0.000 _base.py:297(predict)
      218    0.002    0.000    0.029    0.000 _base.py:287(_decision_function)
        1    0.001    0.001    0.026    0.026 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      218    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      436    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
        2    0.000    0.000    0.005    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
      654    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      218    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    12862    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      436    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      654    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         94026 function calls in 0.104 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.105    0.105 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.105    0.105 surrogate.py:1721(__call__)
        1    0.002    0.002    0.104    0.104 surrogate.py:934(__call__)
       11    0.000    0.000    0.095    0.009 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.095    0.009 surrogate.py:417(__call__)
       19    0.000    0.000    0.094    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.085    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.078    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.078    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.030    0.000 _base.py:297(predict)
      218    0.001    0.000    0.029    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1308    0.004    0.000    0.012    0.000 validation.py:371(_num_samples)
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      218    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      436    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      654    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      218    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    12862    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
     1308    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      436    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      654    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      436    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
     1744    0.001    0.000    0.003    0.000 _config.py:35(get_config)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      436    0.000    0.000    0.003    0.000 _base.py:711(__sklearn_tags__)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         94030 function calls in 0.153 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.154    0.154 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.002    0.002    0.154    0.154 surrogate.py:1721(__call__)
        1    0.002    0.002    0.146    0.146 surrogate.py:934(__call__)
       11    0.000    0.000    0.095    0.009 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.095    0.009 surrogate.py:417(__call__)
       19    0.000    0.000    0.095    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.085    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.078    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.078    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
        1    0.002    0.002    0.048    0.048 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.030    0.000 _base.py:297(predict)
      218    0.002    0.000    0.030    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
        1    0.000    0.000    0.020    0.020 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.019    0.019    0.020    0.020 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.016    0.016    0.016    0.016 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1308    0.004    0.000    0.012    0.000 validation.py:371(_num_samples)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
        2    0.000    0.000    0.009    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.008    0.004    0.009    0.004 spline_interp_Cwrapper.py:50(interpolate)
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      218    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      436    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       18    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
      654    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      218    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    12862    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         94026 function calls in 0.105 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.106    0.106 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.106    0.106 surrogate.py:1721(__call__)
        1    0.002    0.002    0.105    0.105 surrogate.py:934(__call__)
       11    0.000    0.000    0.094    0.009 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.094    0.009 surrogate.py:417(__call__)
       19    0.000    0.000    0.094    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.085    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.078    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.030    0.000 _base.py:297(predict)
      218    0.002    0.000    0.030    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      218    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      436    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      654    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      218    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    12862    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
     1308    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      436    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      654    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      436    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
     1744    0.001    0.000    0.003    0.000 _config.py:35(get_config)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         94006 function calls in 0.105 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.106    0.106 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.106    0.106 surrogate.py:1721(__call__)
        1    0.002    0.002    0.105    0.105 surrogate.py:934(__call__)
       11    0.000    0.000    0.094    0.009 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.094    0.009 surrogate.py:417(__call__)
       19    0.000    0.000    0.094    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.084    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.077    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.029    0.000 _base.py:297(predict)
      218    0.001    0.000    0.029    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      218    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      436    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      654    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      218    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    12862    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
     1308    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      436    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      436    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      654    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         94006 function calls in 0.355 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.356    0.356 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.021    0.021    0.356    0.356 surrogate.py:1721(__call__)
        1    0.003    0.003    0.323    0.323 surrogate.py:934(__call__)
        1    0.009    0.009    0.225    0.225 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.095    0.009 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.095    0.009 surrogate.py:417(__call__)
       19    0.000    0.000    0.094    0.005 surrogate.py:292(__call__)
        1    0.000    0.000    0.092    0.092 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.092    0.092    0.092    0.092 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.084    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.084    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.077    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.070    0.070    0.070    0.070 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      436    0.002    0.000    0.047    0.000 validation.py:2793(validate_data)
        2    0.000    0.000    0.046    0.023 surrogate.py:91(_splinterp_Cwrapper)
        2    0.046    0.023    0.046    0.023 spline_interp_Cwrapper.py:50(interpolate)
      218    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.038    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.029    0.000 _base.py:297(predict)
      218    0.002    0.000    0.029    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
        7    0.012    0.002    0.012    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      218    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      436    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      654    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
      218    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      654    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    12862    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         94006 function calls in 0.104 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.105    0.105 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.000    0.000    0.105    0.105 surrogate.py:1721(__call__)
        1    0.002    0.002    0.104    0.104 surrogate.py:934(__call__)
       11    0.000    0.000    0.096    0.009 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.096    0.009 surrogate.py:417(__call__)
       19    0.000    0.000    0.095    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.086    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.086    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.079    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.079    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.078    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.030    0.000 _base.py:297(predict)
      218    0.002    0.000    0.030    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      436    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      654    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      218    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      654    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    12862    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      436    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      436    0.001    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
     1308    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      654    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     1744    0.001    0.000    0.003    0.000 _config.py:35(get_config)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      436    0.000    0.000    0.003    0.000 _base.py:711(__sklearn_tags__)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         94006 function calls in 0.156 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.157    0.157 benchmark_surrogate_evaluations.py:312(evaluate_case)
        1    0.002    0.002    0.157    0.157 surrogate.py:1721(__call__)
        1    0.002    0.002    0.153    0.153 surrogate.py:934(__call__)
       11    0.000    0.000    0.095    0.009 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.095    0.009 surrogate.py:417(__call__)
       19    0.000    0.000    0.094    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.085    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.078    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.002    0.002    0.055    0.055 surrogate.py:742(_coorbital_to_inertial_frame)
      436    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.030    0.000 _base.py:297(predict)
      218    0.001    0.000    0.029    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
        1    0.019    0.019    0.019    0.019 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      218    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      436    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      654    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      218    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
    12862    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      654    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      436    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
```
