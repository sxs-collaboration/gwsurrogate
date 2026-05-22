# GWSurrogate Evaluation Timing

Generated: 2026-05-22T15:28:56.873128+00:00

Times below are seconds per model evaluation. Raw repeats and context are in the JSON output.

PNG timing table: `test/benchmark_surrogate_evaluations.png`

## Summary

### master

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0192557` s, median `0.0201022` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0195463` s, median `0.0198754` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0206402` s, median `0.0208718` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0202777` s, median `0.0205519` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0171571` s, median `0.0172404` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0167012` s, median `0.016775` s
- `dt=0.5 M`, `f_low=0`: best `0.012874` s, median `0.0129571` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0157605` s, median `0.0158685` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0248669` s, median `0.0248931` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0243326` s, median `0.0243943` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.02628` s, median `0.0263591` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0250007` s, median `0.0250442` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0285629` s, median `0.0286503` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0261833` s, median `0.0267895` s
- `dt=0.5 M`, `f_low=0`: best `0.0216521` s, median `0.0216988` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0247913` s, median `0.0249045` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0581464` s, median `0.0583436` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.024779` s, median `0.0249957` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.10062` s, median `0.100942` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0260093` s, median `0.0262853` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0267674` s, median `0.0272619` s
- `dt=0.1 M`, `f_low=0.002`: best `0.30869` s, median `0.310483` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0239344` s, median `0.0239608` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0825532` s, median `0.0827477` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0233502` s, median `0.0234936` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.00777652` s, median `0.00802234` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0390577` s, median `0.0391717` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.00887235` s, median `0.00924782` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.00978571` s, median `0.0098802` s
- `dt=0.1 M`, `f_low=0.002`: best `0.188503` s, median `0.188779` s
- `dt=0.5 M`, `f_low=0.01`: best `0.00703534` s, median `0.00730336` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0414591` s, median `0.0416856` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0531688` s, median `0.0536661` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0292115` s, median `0.029398` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0799825` s, median `0.0802677` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0308612` s, median `0.0312151` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0314706` s, median `0.0318991` s
- `dt=0.1 M`, `f_low=0.002`: best `0.293017` s, median `0.293772` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0280255` s, median `0.0286295` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0838524` s, median `0.0841591` s

### PR-75

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0168673` s, median `0.0179201` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0175271` s, median `0.0176177` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0186028` s, median `0.0188323` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0175828` s, median `0.0178193` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0147438` s, median `0.0150991` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0142342` s, median `0.0144913` s
- `dt=0.5 M`, `f_low=0`: best `0.0106221` s, median `0.0108076` s
- `dt=0.5 M`, `f_low=0.01`: best `0.013014` s, median `0.0131697` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0191017` s, median `0.019356` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0187133` s, median `0.0188468` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0207386` s, median `0.0208111` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0191995` s, median `0.0195526` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0233378` s, median `0.0235334` s
- `dt=0.1 M`, `f_low=0.01`: best `0.020217` s, median `0.0203517` s
- `dt=0.5 M`, `f_low=0`: best `0.0162834` s, median `0.0164599` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0189357` s, median `0.0190071` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0581926` s, median `0.0584925` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.025209` s, median `0.0254212` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0979792` s, median `0.0987033` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0267961` s, median `0.0271881` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0271467` s, median `0.0272322` s
- `dt=0.1 M`, `f_low=0.002`: best `0.299156` s, median `0.299918` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0245388` s, median `0.0247908` s
- `dt=0.5 M`, `f_low=0.002`: best `0.080735` s, median `0.0809172` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0219122` s, median `0.0219925` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.00720545` s, median `0.00744115` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0364041` s, median `0.0364986` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0085435` s, median `0.00866131` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.00909432` s, median `0.00929528` s
- `dt=0.1 M`, `f_low=0.002`: best `0.173655` s, median `0.173776` s
- `dt=0.5 M`, `f_low=0.01`: best `0.00701448` s, median `0.00716414` s
- `dt=0.5 M`, `f_low=0.002`: best `0.038169` s, median `0.0383083` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0521348` s, median `0.0525111` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0290664` s, median `0.0294954` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0779458` s, median `0.0783415` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0300752` s, median `0.0306763` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.030674` s, median `0.0311576` s
- `dt=0.1 M`, `f_low=0.002`: best `0.285631` s, median `0.286162` s
- `dt=0.5 M`, `f_low=0.01`: best `0.027967` s, median `0.028166` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0807724` s, median `0.0809142` s

## Context

### master

- Git branch: `master`
- Git commit: `0cce25ea4b06acdf5bafa6bce93505f987588890`
- Git describe: `v1.1.8-34-g0cce25e`
- Python: `3.14.5 (main, May 11 2026, 02:45:53) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1013-azure x86_64`
- CPU count: `4`
- Conda env: `unknown`

Submodules:

- `gwsurrogate/eval_pysur`: `6a51ecba0ed8ddc26a85e5d2918596aa9f58f534` initialized ((heads/master))

### PR-75

- Git branch: `unknown`
- Git commit: `d65c7efbc78cb04573fc68dbc5ef438c7debb7db`
- Git describe: `v1.1.8-41-gd65c7ef`
- Python: `3.14.5 (main, May 11 2026, 02:45:53) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1013-azure x86_64`
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
BogoMIPS:                                5192.27
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

#### PR-75

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
BogoMIPS:                                5192.27
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
         26052 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:358(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:592(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:150(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:282(get_time_deriv)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:735(__call__)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:76(rotateWaveform)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:568(_one_backward_RK4_step)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:772(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:626(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:493(_initialize)
      839    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:311(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:305(get_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         26053 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:358(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:592(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:150(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:282(get_time_deriv)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:813(inertial_waveform_modes)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:76(rotateWaveform)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:735(__call__)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:568(_one_backward_RK4_step)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:772(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     2806    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:626(_integrate_backward)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.001    0.001    0.001    0.001 {method 'update' of 'dict' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:493(_initialize)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:311(_get_t_from_omega)
      839    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:305(get_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
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
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:358(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:592(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:150(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:282(get_time_deriv)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:735(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:76(rotateWaveform)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:568(_one_backward_RK4_step)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:772(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:626(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:493(_initialize)
      839    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:311(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:305(get_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         26053 function calls in 0.029 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.029    0.029 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:358(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:592(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:150(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:282(get_time_deriv)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:735(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:76(rotateWaveform)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:568(_one_backward_RK4_step)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:772(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     2806    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:626(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:493(_initialize)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      839    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:311(_get_t_from_omega)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:305(get_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         19278 function calls in 0.025 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.025    0.025 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.025    0.025 surrogate.py:1721(__call__)
        1    0.000    0.000    0.025    0.025 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:358(__call__)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:592(_integrate_forward)
      238    0.001    0.000    0.008    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
        4    0.000    0.000    0.007    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
      714    0.001    0.000    0.006    0.000 precessing_surrogate.py:150(_eval_vector_fit)
        1    0.000    0.000    0.006    0.006 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.006    0.006    0.006    0.006 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:735(__call__)
     2390    0.001    0.000    0.005    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:772(_eval_comp)
     2390    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:76(rotateWaveform)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initial_RK4)
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      716    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      534    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:674(_assemble_mode_pair)
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:798(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:808(coorb_spins_from_copr_spins)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         26659 function calls in 0.026 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.026    0.026 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.026    0.026 surrogate.py:1721(__call__)
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:358(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:150(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:282(get_time_deriv)
     2876    0.002    0.000    0.006    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:568(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:735(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:772(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:592(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     2876    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:76(rotateWaveform)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:311(_get_t_from_omega)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:305(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:493(_initialize)
      862    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      839    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         19278 function calls in 0.020 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.020    0.020 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.020    0.020 surrogate.py:1721(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:358(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:592(_integrate_forward)
      238    0.001    0.000    0.008    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:150(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:735(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:772(_eval_comp)
     2390    0.001    0.000    0.005    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
     2390    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:76(rotateWaveform)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:527(_initial_RK4)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      716    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      534    0.000    0.000    0.000    0.000 {built-in method numpy.array}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:674(_assemble_mode_pair)
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:798(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:808(coorb_spins_from_copr_spins)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         26659 function calls in 0.025 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.025    0.025 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.025    0.025 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:358(__call__)
      279    0.001    0.000    0.009    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:150(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:282(get_time_deriv)
     2876    0.002    0.000    0.006    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:735(__call__)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:568(_one_backward_RK4_step)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:772(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:592(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     2876    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:76(rotateWaveform)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:311(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:305(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:493(_initialize)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      862    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      839    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         41113 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.037    0.037 surrogate.py:1721(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:1231(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:358(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:592(_integrate_forward)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:150(_eval_vector_fit)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:282(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:568(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:626(_integrate_backward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:76(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1157(<lambda>)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:735(__call__)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_comp)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:311(_get_t_from_omega)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:305(get_omega)
     1640    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:493(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      963    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         41114 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.037    0.037 surrogate.py:1721(__call__)
        1    0.000    0.000    0.036    0.036 precessing_surrogate.py:1231(__call__)
        1    0.000    0.000    0.036    0.036 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:358(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:592(_integrate_forward)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:150(_eval_vector_fit)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:282(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:568(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:626(_integrate_backward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:76(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1157(<lambda>)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:735(__call__)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_comp)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:311(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:305(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     1640    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:493(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      963    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         41113 function calls in 0.039 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.039    0.039 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.039    0.039 surrogate.py:1721(__call__)
        1    0.000    0.000    0.038    0.038 precessing_surrogate.py:1231(__call__)
        1    0.000    0.000    0.038    0.038 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:358(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:592(_integrate_forward)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:150(_eval_vector_fit)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:282(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:568(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:626(_integrate_backward)
        4    0.000    0.000    0.005    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:76(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1157(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:735(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_comp)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:311(_get_t_from_omega)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:305(get_omega)
     1640    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:493(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      963    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         41114 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1721(__call__)
        1    0.000    0.000    0.038    0.038 precessing_surrogate.py:1231(__call__)
        1    0.000    0.000    0.038    0.038 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:358(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:150(_eval_vector_fit)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:592(_integrate_forward)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:282(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:568(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:626(_integrate_backward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:76(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1157(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:735(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_comp)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:311(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:305(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     1640    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:493(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      963    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         33092 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 90 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1721(__call__)
        1    0.000    0.000    0.038    0.038 precessing_surrogate.py:1231(__call__)
        1    0.000    0.000    0.038    0.038 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:358(__call__)
        1    0.001    0.001    0.020    0.020 precessing_surrogate.py:592(_integrate_forward)
      505    0.001    0.000    0.019    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
     1515    0.003    0.000    0.015    0.000 precessing_surrogate.py:150(_eval_vector_fit)
     4637    0.003    0.000    0.012    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
        4    0.000    0.000    0.011    0.003 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:76(rotateWaveform)
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1157(<lambda>)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:735(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_comp)
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     1517    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:527(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      623    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:798(rotate_spin)
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:808(coorb_spins_from_copr_spins)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:830(normalize_spin)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         42469 function calls in 0.039 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.039    0.039 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.039    0.039 surrogate.py:1721(__call__)
        1    0.000    0.000    0.039    0.039 precessing_surrogate.py:1231(__call__)
        1    0.000    0.000    0.039    0.039 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:358(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:150(_eval_vector_fit)
     5274    0.003    0.000    0.014    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:626(_integrate_backward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:592(_integrate_forward)
     5274    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:282(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:568(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:76(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1157(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:735(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:311(_get_t_from_omega)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:305(get_omega)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_comp)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
     1640    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:493(_initialize)
     1102    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         33092 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 90 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:1231(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:358(__call__)
        1    0.001    0.001    0.020    0.020 precessing_surrogate.py:592(_integrate_forward)
      505    0.001    0.000    0.019    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
     1515    0.003    0.000    0.015    0.000 precessing_surrogate.py:150(_eval_vector_fit)
     4637    0.003    0.000    0.012    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
     4637    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:76(rotateWaveform)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1157(<lambda>)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:735(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_comp)
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     1517    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:527(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      623    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:798(rotate_spin)
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:808(coorb_spins_from_copr_spins)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         42469 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1721(__call__)
        1    0.000    0.000    0.038    0.038 precessing_surrogate.py:1231(__call__)
        1    0.000    0.000    0.038    0.038 precessing_surrogate.py:903(__call__)
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:358(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:264(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:150(_eval_vector_fit)
     5274    0.003    0.000    0.014    0.000 precessing_surrogate.py:123(_eval_scalar_fit)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:626(_integrate_backward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:592(_integrate_forward)
     5274    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:282(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:568(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:813(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:76(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1157(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:735(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:311(_get_t_from_omega)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_comp)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:305(get_omega)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1640    0.001    0.000    0.001    0.000 {built-in method numpy.asarray}
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:820(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:493(_initialize)
     1102    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         9044 function calls in 0.061 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.061    0.061 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.061    0.061 surrogate.py:1721(__call__)
        1    0.003    0.003    0.057    0.057 surrogate.py:934(__call__)
        1    0.001    0.001    0.033    0.033 surrogate.py:742(_coorbital_to_inertial_frame)
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
        2    0.000    0.000    0.006    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.006    0.003    0.006    0.003 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       21    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        3    0.001    0.000    0.001    0.000 {built-in method numpy.ascontiguousarray}
        9    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
       26    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         9040 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.002    0.002    0.030    0.030 surrogate.py:934(__call__)
       12    0.000    0.000    0.021    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.021    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.020    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.011    0.000 nodeFunction.py:125(__call__)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         9044 function calls in 0.102 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.102    0.102 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.102    0.102 surrogate.py:1721(__call__)
        1    0.003    0.003    0.082    0.082 surrogate.py:934(__call__)
        1    0.002    0.002    0.058    0.058 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.020    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.020    0.001 surrogate.py:292(__call__)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.016    0.016 {method 'update' of 'dict' objects}
       21    0.016    0.001    0.016    0.001 surrogate.py:2126(<genexpr>)
        2    0.000    0.000    0.012    0.006 surrogate.py:91(_splinterp_Cwrapper)
        2    0.012    0.006    0.012    0.006 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.002    0.001    0.002    0.001 _function_base_impl.py:1402(diff)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
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
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         9040 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.002    0.002    0.030    0.030 surrogate.py:934(__call__)
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.018    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.011    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.011    0.000 nodeFunction.py:125(__call__)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.005    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
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
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.002    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         9017 function calls in 0.312 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.312    0.312 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.024    0.024    0.312    0.312 surrogate.py:1721(__call__)
        1    0.003    0.003    0.275    0.275 surrogate.py:934(__call__)
        1    0.009    0.009    0.251    0.251 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.100    0.100 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.099    0.099    0.100    0.100 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.074    0.074    0.074    0.074 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.060    0.030 surrogate.py:91(_splinterp_Cwrapper)
        2    0.059    0.029    0.060    0.030 spline_interp_Cwrapper.py:50(interpolate)
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.001    0.000    0.020    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
        9    0.013    0.001    0.013    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.005    0.001    0.005    0.001 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
        6    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        2    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         9017 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.002    0.002    0.028    0.028 surrogate.py:934(__call__)
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         9017 function calls in 0.086 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.086    0.086 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.086    0.086 surrogate.py:1721(__call__)
        1    0.002    0.002    0.082    0.082 surrogate.py:934(__call__)
        1    0.002    0.002    0.060    0.060 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.025    0.025 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.024    0.024    0.025    0.025 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.020    0.020    0.020    0.020 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
        2    0.000    0.000    0.012    0.006 surrogate.py:91(_splinterp_Cwrapper)
        2    0.012    0.006    0.012    0.006 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         3540 function calls in 0.025 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.025    0.025 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.025    0.025 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 surrogate.py:934(__call__)
        1    0.001    0.001    0.018    0.018 surrogate.py:742(_coorbital_to_inertial_frame)
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.003    0.005    0.003 spline_interp_Cwrapper.py:50(interpolate)
        1    0.005    0.005    0.005    0.005 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.005    0.005 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.004    0.004    0.005    0.005 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
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
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         3536 function calls in 0.009 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.009    0.009 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.009    0.009 surrogate.py:1721(__call__)
        1    0.000    0.000    0.009    0.009 surrogate.py:934(__call__)
        6    0.000    0.000    0.004    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.004    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         3540 function calls in 0.041 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.041    0.041 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.041    0.041 surrogate.py:1721(__call__)
        1    0.000    0.000    0.037    0.037 surrogate.py:934(__call__)
        1    0.001    0.001    0.031    0.031 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.011    0.011    0.011    0.011 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
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
        1    0.000    0.000    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
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
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
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
        1    0.000    0.000    0.011    0.011 surrogate.py:934(__call__)
        1    0.000    0.000    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.002    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         3523 function calls in 0.191 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.191    0.191 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.012    0.012    0.191    0.191 surrogate.py:1721(__call__)
        1    0.000    0.000    0.171    0.171 surrogate.py:934(__call__)
        1    0.007    0.007    0.166    0.166 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.061    0.061    0.061    0.061 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.053    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.051    0.026    0.053    0.026 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.041    0.041 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.041    0.041    0.041    0.041 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        5    0.007    0.001    0.007    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       20    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
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
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         3523 function calls in 0.043 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.043    0.043 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.043    0.043 surrogate.py:1721(__call__)
        1    0.000    0.000    0.041    0.041 surrogate.py:934(__call__)
        1    0.001    0.001    0.036    0.036 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.012    0.012    0.012    0.012 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.011    0.006 surrogate.py:91(_splinterp_Cwrapper)
        2    0.011    0.005    0.011    0.006 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.009    0.009 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.009    0.009 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         10972 function calls in 0.058 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.058    0.058 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.058    0.058 surrogate.py:1721(__call__)
        1    0.002    0.002    0.054    0.054 surrogate.py:934(__call__)
        1    0.001    0.001    0.026    0.026 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.006    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.003    0.006    0.003 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       18    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         10968 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1721(__call__)
        1    0.002    0.002    0.034    0.034 surrogate.py:934(__call__)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         10972 function calls in 0.087 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.086    0.086 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.086    0.086 surrogate.py:1721(__call__)
        1    0.002    0.002    0.079    0.079 surrogate.py:934(__call__)
        1    0.002    0.002    0.050    0.050 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.020    0.020 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.019    0.019    0.020    0.020 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.017    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
        1    0.016    0.016    0.016    0.016 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        2    0.010    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       18    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
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
      218    0.000    0.000    0.017    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
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
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         10948 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

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
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.002    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         10948 function calls in 0.296 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.296    0.296 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.022    0.022    0.296    0.296 surrogate.py:1721(__call__)
        1    0.003    0.003    0.265    0.265 surrogate.py:934(__call__)
        1    0.009    0.009    0.236    0.236 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.091    0.091 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.090    0.090    0.091    0.091 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.070    0.070    0.070    0.070 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.056    0.028 surrogate.py:91(_splinterp_Cwrapper)
        2    0.056    0.028    0.056    0.028 spline_interp_Cwrapper.py:50(interpolate)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.017    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.017    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        7    0.010    0.001    0.010    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.005    0.001    0.005    0.001 _function_base_impl.py:1402(diff)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         10948 function calls in 0.033 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.033    0.033 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.033    0.033 surrogate.py:1721(__call__)
        1    0.002    0.002    0.033    0.033 surrogate.py:934(__call__)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         10948 function calls in 0.090 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.090    0.090 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.090    0.090 surrogate.py:1721(__call__)
        1    0.002    0.002    0.086    0.086 surrogate.py:934(__call__)
        1    0.002    0.002    0.057    0.057 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.019    0.019    0.019    0.019 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.017    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.012    0.006 surrogate.py:91(_splinterp_Cwrapper)
        2    0.012    0.006    0.012    0.006 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

#### PR-75

##### NRSur7dq4 / mks_dt_0.000244140625_flow_0

```text
         14341 function calls in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.022    0.022 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.022    0.022 surrogate.py:1721(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:362(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:743(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:780(_eval_comp)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:286(get_time_deriv)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:572(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:596(_integrate_forward)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:630(_integrate_backward)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
      295    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:497(_initialize)
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:315(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:309(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:806(rotate_spin)
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         14342 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:362(__call__)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:743(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:780(_eval_comp)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:286(get_time_deriv)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:572(_one_backward_RK4_step)
      279    0.000    0.000    0.004    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:596(_integrate_forward)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:630(_integrate_backward)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      295    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:497(_initialize)
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:315(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:309(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         14341 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:362(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:743(__call__)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:780(_eval_comp)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:286(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:572(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:596(_integrate_forward)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:630(_integrate_backward)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
      295    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      792    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:497(_initialize)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:315(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:309(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         14342 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:362(__call__)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:743(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:780(_eval_comp)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:286(get_time_deriv)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:572(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:596(_integrate_forward)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:630(_integrate_backward)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      295    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:497(_initialize)
      792    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:315(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:309(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         9295 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 85 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1721(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:911(__call__)
        4    0.000    0.000    0.006    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.006    0.006 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.006    0.006    0.006    0.006 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:743(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:780(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:362(__call__)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:596(_integrate_forward)
      238    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
      238    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      486    0.001    0.000    0.001    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      248    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      534    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:531(_initial_RK4)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:806(rotate_spin)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:678(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:816(coorb_spins_from_copr_spins)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:838(normalize_spin)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      504    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         14878 function calls in 0.020 seconds

   Ordered by: cumulative time
   List reduced from 97 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.020    0.020 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.020    0.020 surrogate.py:1721(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:362(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:743(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:780(_eval_comp)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:286(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:572(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:821(inertial_waveform_modes)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:596(_integrate_forward)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:630(_integrate_backward)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:315(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:309(get_omega)
      365    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      365    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:497(_initialize)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      862    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         9295 function calls in 0.014 seconds

   Ordered by: cumulative time
   List reduced from 85 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.014    0.014 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.014    0.014 surrogate.py:1721(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:743(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:780(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:362(__call__)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:596(_integrate_forward)
      238    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      238    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      486    0.001    0.000    0.001    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      248    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      534    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:531(_initial_RK4)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:806(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:816(coorb_spins_from_copr_spins)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:678(_assemble_mode_pair)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      504    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         14878 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 97 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1721(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:362(__call__)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:743(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:780(_eval_comp)
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:286(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:572(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:821(inertial_waveform_modes)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:596(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:630(_integrate_backward)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:315(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:309(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
      365    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:497(_initialize)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      862    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1006    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         13382 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:1239(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:362(__call__)
      546    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:596(_integrate_forward)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:286(get_time_deriv)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:572(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:743(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:780(_eval_comp)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:630(_integrate_backward)
      767    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:315(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:309(get_omega)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      221    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      963    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:497(_initialize)
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      191    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         13383 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.023    0.023 surrogate.py:1721(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:1239(__call__)
        1    0.000    0.000    0.023    0.023 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:362(__call__)
      546    0.001    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:596(_integrate_forward)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:286(get_time_deriv)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:572(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:743(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:780(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:630(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:315(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:309(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      221    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:497(_initialize)
      963    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      191    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         13382 function calls in 0.026 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.026    0.026 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.026    0.026 surrogate.py:1721(__call__)
        1    0.000    0.000    0.025    0.025 precessing_surrogate.py:1239(__call__)
        1    0.000    0.000    0.025    0.025 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:362(__call__)
      546    0.001    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:596(_integrate_forward)
      546    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:286(get_time_deriv)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:572(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:743(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:780(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:630(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:315(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:309(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      221    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:497(_initialize)
      963    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      191    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         13383 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:1239(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:362(__call__)
      546    0.001    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:596(_integrate_forward)
      546    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.004    0.000 precessing_surrogate.py:286(get_time_deriv)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:572(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:743(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:780(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:630(_integrate_backward)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:315(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:309(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      221    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:497(_initialize)
      963    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      221    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      191    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         7669 function calls in 0.026 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.026    0.026 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.026    0.026 surrogate.py:1721(__call__)
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:1239(__call__)
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:911(__call__)
        4    0.000    0.000    0.011    0.003 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:362(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:596(_integrate_forward)
      505    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      505    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:743(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:780(_eval_comp)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      623    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       92    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:806(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:816(coorb_spins_from_copr_spins)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:531(_initial_RK4)
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:838(normalize_spin)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.000    0.000 fromnumeric.py:2304(sum)
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
        1    0.000    0.000    0.025    0.025 precessing_surrogate.py:1239(__call__)
        1    0.000    0.000    0.025    0.025 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:362(__call__)
      546    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      546    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:596(_integrate_forward)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:286(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:572(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:630(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:743(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:780(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:315(_get_t_from_omega)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:309(get_omega)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      360    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
      360    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
     1102    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:497(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      191    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         7669 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1721(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:1239(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:362(__call__)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:596(_integrate_forward)
      505    0.000    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      505    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:743(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:780(_eval_comp)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      623    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       92    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:806(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:531(_initial_RK4)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:816(coorb_spins_from_copr_spins)
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:838(normalize_spin)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        6    0.000    0.000    0.000    0.000 fromnumeric.py:2304(sum)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         14460 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:1239(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:911(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:362(__call__)
      546    0.001    0.000    0.007    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
      546    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:596(_integrate_forward)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:286(get_time_deriv)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:821(inertial_waveform_modes)
        1    0.002    0.002    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:572(_one_backward_RK4_step)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:630(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:743(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:780(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:315(_get_t_from_omega)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:309(get_omega)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      360    0.000    0.000    0.001    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:828(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      360    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:497(_initialize)
     1102    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      191    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:806(rotate_spin)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         9044 function calls in 0.062 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.062    0.062 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.061    0.061 surrogate.py:1721(__call__)
        1    0.003    0.003    0.056    0.056 surrogate.py:934(__call__)
        1    0.001    0.001    0.032    0.032 surrogate.py:742(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.021    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.020    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.020    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.015    0.015 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.013    0.013    0.015    0.015 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.005    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 {method 'update' of 'dict' objects}
       21    0.003    0.000    0.003    0.000 surrogate.py:2126(<genexpr>)
      177    0.001    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
        3    0.001    0.000    0.001    0.000 {built-in method numpy.ascontiguousarray}
        9    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

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
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         9044 function calls in 0.100 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.100    0.100 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.100    0.100 surrogate.py:1721(__call__)
        1    0.004    0.004    0.081    0.081 surrogate.py:934(__call__)
        1    0.002    0.002    0.056    0.056 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.023    0.023    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.021    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.021    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.021    0.001 surrogate.py:292(__call__)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.014    0.014 {method 'update' of 'dict' objects}
       21    0.014    0.001    0.014    0.001 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.009    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.009    0.005 spline_interp_Cwrapper.py:50(interpolate)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.003    0.001    0.003    0.001 _function_base_impl.py:1402(diff)
        9    0.003    0.000    0.003    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         9040 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

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
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         9017 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.002    0.002    0.031    0.031 surrogate.py:934(__call__)
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.011    0.000 nodeFunction.py:125(__call__)
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         9017 function calls in 0.303 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.303    0.303 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.026    0.026    0.303    0.303 surrogate.py:1721(__call__)
        1    0.003    0.003    0.264    0.264 surrogate.py:934(__call__)
        1    0.009    0.009    0.240    0.240 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.102    0.102 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.101    0.101    0.102    0.102 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.074    0.074    0.074    0.074 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.047    0.024 surrogate.py:91(_splinterp_Cwrapper)
        2    0.046    0.023    0.047    0.024 spline_interp_Cwrapper.py:50(interpolate)
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.001    0.000    0.020    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
        9    0.013    0.001    0.013    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        6    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
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
        1    0.002    0.002    0.029    0.029 surrogate.py:934(__call__)
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.020    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.020    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
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
        1    0.002    0.002    0.080    0.080 surrogate.py:934(__call__)
        1    0.002    0.002    0.058    0.058 surrogate.py:742(_coorbital_to_inertial_frame)
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
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
      177    0.001    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
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
        1    0.001    0.001    0.016    0.016 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.005    0.005    0.005    0.005 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.005    0.005 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.004    0.004    0.005    0.005 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
        2    0.000    0.000    0.004    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.004    0.002 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
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
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
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
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         3540 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.038    0.038 surrogate.py:1721(__call__)
        1    0.000    0.000    0.034    0.034 surrogate.py:934(__call__)
        1    0.001    0.001    0.029    0.029 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.011    0.011    0.011    0.011 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.007    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.007    0.003    0.007    0.003 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         3536 function calls in 0.011 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1721(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:934(__call__)
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
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
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
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         3523 function calls in 0.175 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.175    0.175 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.012    0.012    0.175    0.175 surrogate.py:1721(__call__)
        1    0.000    0.000    0.156    0.156 surrogate.py:934(__call__)
        1    0.007    0.007    0.151    0.151 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.061    0.061    0.061    0.061 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.041    0.041 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.041    0.041    0.041    0.041 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.037    0.019 surrogate.py:91(_splinterp_Cwrapper)
        2    0.036    0.018    0.037    0.019 spline_interp_Cwrapper.py:50(interpolate)
        5    0.007    0.001    0.007    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1402(diff)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       20    0.001    0.000    0.001    0.000 _internal.py:263(__init__)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
        2    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
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
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
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
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         10972 function calls in 0.058 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.057    0.057 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.057    0.057 surrogate.py:1721(__call__)
        1    0.002    0.002    0.053    0.053 surrogate.py:934(__call__)
        1    0.001    0.001    0.025    0.025 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.005    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       18    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         10968 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1721(__call__)
        1    0.002    0.002    0.034    0.034 surrogate.py:934(__call__)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         10972 function calls in 0.083 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.083    0.083 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.083    0.083 surrogate.py:1721(__call__)
        1    0.002    0.002    0.075    0.075 surrogate.py:934(__call__)
        1    0.002    0.002    0.048    0.048 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.020    0.020 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.019    0.019    0.020    0.020 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.016    0.016    0.016    0.016 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.008    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       18    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         10968 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1721(__call__)
        1    0.002    0.002    0.034    0.034 surrogate.py:934(__call__)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
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
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
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
        1    0.002    0.002    0.036    0.036 surrogate.py:934(__call__)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         10948 function calls in 0.289 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.289    0.289 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.022    0.022    0.289    0.289 surrogate.py:1721(__call__)
        1    0.003    0.003    0.256    0.256 surrogate.py:934(__call__)
        1    0.008    0.008    0.228    0.228 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.093    0.093 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.092    0.092    0.093    0.093 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.070    0.070    0.070    0.070 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.048    0.024 surrogate.py:91(_splinterp_Cwrapper)
        2    0.048    0.024    0.048    0.024 spline_interp_Cwrapper.py:50(interpolate)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
        7    0.011    0.002    0.011    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.005    0.001    0.005    0.001 _function_base_impl.py:1402(diff)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         10948 function calls in 0.033 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.033    0.033 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.033    0.033 surrogate.py:1721(__call__)
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
        1    0.001    0.001    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         10948 function calls in 0.086 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.086    0.086 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.086    0.086 surrogate.py:1721(__call__)
        1    0.002    0.002    0.083    0.083 surrogate.py:934(__call__)
        1    0.002    0.002    0.055    0.055 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.025    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.025    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.022    0.022 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.022    0.022 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.019    0.019    0.019    0.019 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
```
