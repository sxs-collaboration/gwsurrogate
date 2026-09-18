# GWSurrogate Evaluation Timing

Generated: 2026-09-18T23:45:09.797951+00:00

Times below are seconds per model evaluation. Raw repeats and context are in the JSON output.

PNG timing table: `test/benchmark_surrogate_evaluations.png`

## Summary

### master

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0148122` s, median `0.0149321` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.014503` s, median `0.0145636` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0155061` s, median `0.0156123` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.015074` s, median `0.0152353` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0121261` s, median `0.0122608` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0120399` s, median `0.012327` s
- `dt=0.5 M`, `f_low=0`: best `0.00840076` s, median `0.00852638` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0113374` s, median `0.0114159` s

#### NRSur7dq4v2

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0229531` s, median `0.0229851` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0226542` s, median `0.0227148` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0236315` s, median `0.0237326` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0231595` s, median `0.0232358` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0340983` s, median `0.0346333` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0323073` s, median `0.0325137` s
- `dt=0.5 M`, `f_low=0`: best `0.0283901` s, median `0.0285046` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0311668` s, median `0.031333` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0134239` s, median `0.0134733` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0129441` s, median `0.0129999` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0142883` s, median `0.0143501` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0133875` s, median `0.0134666` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0193011` s, median `0.0194156` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0165195` s, median `0.016565` s
- `dt=0.5 M`, `f_low=0`: best `0.012549` s, median `0.0128595` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0152503` s, median `0.0156372` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0625039` s, median `0.0628151` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0271173` s, median `0.0274849` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.101199` s, median `0.101431` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0283918` s, median `0.0285724` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0290354` s, median `0.0292729` s
- `dt=0.1 M`, `f_low=0.002`: best `0.304618` s, median `0.305689` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0261497` s, median `0.0262465` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0851673` s, median `0.0853164` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0207608` s, median `0.0208585` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.00722744` s, median `0.00753913` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0343069` s, median `0.034419` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.00813937` s, median `0.00822459` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.00825342` s, median `0.00837079` s
- `dt=0.1 M`, `f_low=0.002`: best `0.162489` s, median `0.163086` s
- `dt=0.5 M`, `f_low=0.01`: best `0.00629109` s, median `0.00672529` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0358674` s, median `0.0360709` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0536593` s, median `0.0542362` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0305991` s, median `0.0306821` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0807279` s, median `0.0810497` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0318246` s, median `0.0319759` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0320292` s, median `0.0323295` s
- `dt=0.1 M`, `f_low=0.002`: best `0.285314` s, median `0.286313` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0290536` s, median `0.0293714` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0838329` s, median `0.0840917` s

### PR-90

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0132327` s, median `0.0134109` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0124895` s, median `0.0125703` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0139371` s, median `0.0140533` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0133737` s, median `0.0134272` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0118335` s, median `0.011882` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0114665` s, median `0.0116384` s
- `dt=0.5 M`, `f_low=0`: best `0.00786298` s, median `0.00792207` s
- `dt=0.5 M`, `f_low=0.01`: best `0.010679` s, median `0.0107686` s

#### NRSur7dq4v2

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0225486` s, median `0.0227323` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0222264` s, median `0.0226411` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0233157` s, median `0.0234488` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0228632` s, median `0.0229979` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0336487` s, median `0.0339107` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0320938` s, median `0.032501` s
- `dt=0.5 M`, `f_low=0`: best `0.0281164` s, median `0.0285875` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0307075` s, median `0.0307823` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0131758` s, median `0.0133017` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.012712` s, median `0.0127524` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0140121` s, median `0.0140824` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0131879` s, median `0.0132074` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0187642` s, median `0.0188539` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0158887` s, median `0.0159498` s
- `dt=0.5 M`, `f_low=0`: best `0.012123` s, median `0.0121988` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0146786` s, median `0.0148222` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0593559` s, median `0.0594149` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0270583` s, median `0.0273572` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.101622` s, median `0.101915` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0283759` s, median `0.0285842` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0285743` s, median `0.0287583` s
- `dt=0.1 M`, `f_low=0.002`: best `0.299833` s, median `0.300142` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0255783` s, median `0.0259145` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0847429` s, median `0.085085` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0204643` s, median `0.0206902` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.00710648` s, median `0.00716459` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0338527` s, median `0.0339847` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.00800832` s, median `0.00814303` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.00822107` s, median `0.0082514` s
- `dt=0.1 M`, `f_low=0.002`: best `0.163706` s, median `0.163826` s
- `dt=0.5 M`, `f_low=0.01`: best `0.00642668` s, median `0.00647168` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0360915` s, median `0.0361569` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.053013` s, median `0.0533389` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0301729` s, median `0.0302671` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0922211` s, median `0.0927716` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0312838` s, median `0.0314545` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0317689` s, median `0.0319127` s
- `dt=0.1 M`, `f_low=0.002`: best `0.279072` s, median `0.279614` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0292366` s, median `0.0294077` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0838994` s, median `0.0842748` s

## Context

### master

- Git branch: `master`
- Git commit: `fe5401769b45cfac626974263a77c5f12657b9bc`
- Git describe: `v1.2.0-5-gfe54017`
- Python: `3.14.7 (main, Aug  6 2026, 02:19:46) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1022-azure x86_64`
- CPU count: `4`
- Conda env: `unknown`

Submodules:

- `gwsurrogate/eval_pysur`: `6a51ecba0ed8ddc26a85e5d2918596aa9f58f534` initialized ((heads/master))

### PR-90

- Git branch: `unknown`
- Git commit: `78bbb327cc57a95e13bc8c7d02899825676ad4bf`
- Git describe: `v1.1.9-74-g78bbb32`
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

#### PR-90

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
         9264 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1722(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:1108(inertial_waveform_modes)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      574    0.001    0.000    0.002    0.000 surrogate.py:2455(get_fit_params)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
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
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      574    0.001    0.000    0.002    0.000 surrogate.py:2455(get_fit_params)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
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
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      254    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
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
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:1108(inertial_waveform_modes)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      574    0.001    0.000    0.002    0.000 surrogate.py:2455(get_fit_params)
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
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2135(<genexpr>)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         9265 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1722(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:1108(inertial_waveform_modes)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:78(rotateWaveform)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.004    0.004    0.004    0.004 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      574    0.001    0.000    0.002    0.000 surrogate.py:2455(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
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
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:320(get_omega)
       22    0.000    0.000    0.000    0.000 surrogate.py:2135(<genexpr>)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
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
      238    0.000    0.000    0.002    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      486    0.001    0.000    0.002    0.000 surrogate.py:2455(get_fit_params)
      238    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      492    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
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
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
       13    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:112(<genexpr>)
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
      644    0.002    0.000    0.002    0.000 surrogate.py:2455(get_fit_params)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      820    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      254    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
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
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      486    0.001    0.000    0.002    0.000 surrogate.py:2455(get_fit_params)
      238    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        3    0.000    0.000    0.000    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      492    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1101(coorb_spins_from_copr_spins)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:1081(rotate_spin)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         9801 function calls in 0.015 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.015    0.015 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.015    0.015 surrogate.py:1722(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
      644    0.001    0.000    0.002    0.000 surrogate.py:2455(get_fit_params)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:912(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      820    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
```

##### NRSur7dq4v2 / mks_dt_0.000244140625_flow_0

```text
         20275 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1722(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:1214(__call__)
        1    0.001    0.001    0.017    0.017 precessing_surrogate.py:1015(__call__)
       84    0.000    0.000    0.017    0.000 precessing_surrogate.py:1061(_eval_comp)
       84    0.004    0.000    0.017    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     2264    0.008    0.000    0.011    0.000 surrogate.py:2568(get_fit_params)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
     4704    0.003    0.000    0.003    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
     1985    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     2264    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
       86    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
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
      356    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### NRSur7dq4v2 / mks_dt_0.000244140625_flow_20

```text
         20276 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1722(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1015(__call__)
       84    0.000    0.000    0.017    0.000 precessing_surrogate.py:1061(_eval_comp)
       84    0.004    0.000    0.016    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     2264    0.007    0.000    0.011    0.000 surrogate.py:2568(get_fit_params)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
     4704    0.003    0.000    0.003    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
     1985    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
     2264    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
       86    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
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
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1015(__call__)
       84    0.000    0.000    0.017    0.000 precessing_surrogate.py:1061(_eval_comp)
       84    0.004    0.000    0.017    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     2264    0.008    0.000    0.011    0.000 surrogate.py:2568(get_fit_params)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
     4704    0.003    0.000    0.003    0.000 {built-in method numpy.array}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
     1985    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
     2264    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
       86    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
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
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2135(<genexpr>)
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
     2264    0.007    0.000    0.011    0.000 surrogate.py:2568(get_fit_params)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
     4704    0.003    0.000    0.003    0.000 {built-in method numpy.array}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
     1985    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     2264    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
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
       22    0.000    0.000    0.000    0.000 surrogate.py:2135(<genexpr>)
```

##### NRSur7dq4v2 / geom_dt_0.1_flow_0

```text
         21680 function calls in 0.044 seconds

   Ordered by: cumulative time
   List reduced from 86 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.044    0.044 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.044    0.044 surrogate.py:1722(__call__)
        1    0.000    0.000    0.043    0.043 precessing_surrogate.py:1214(__call__)
        1    0.001    0.001    0.027    0.027 precessing_surrogate.py:1015(__call__)
      128    0.000    0.000    0.025    0.000 precessing_surrogate.py:1061(_eval_comp)
      128    0.006    0.000    0.025    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     3195    0.010    0.000    0.015    0.000 surrogate.py:2568(get_fit_params)
        4    0.000    0.000    0.009    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
     6396    0.004    0.000    0.004    0.000 {built-in method numpy.array}
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:607(_integrate_forward)
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
      322    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1101(coorb_spins_from_copr_spins)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:1081(rotate_spin)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
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
     3353    0.011    0.000    0.016    0.000 surrogate.py:2568(get_fit_params)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
     6882    0.004    0.000    0.004    0.000 {built-in method numpy.array}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     3074    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     3353    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
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
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4v2 / geom_dt_0.5_flow_0

```text
         21680 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 86 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1722(__call__)
        1    0.000    0.000    0.038    0.038 precessing_surrogate.py:1214(__call__)
        1    0.001    0.001    0.027    0.027 precessing_surrogate.py:1015(__call__)
      128    0.000    0.000    0.026    0.000 precessing_surrogate.py:1061(_eval_comp)
      128    0.006    0.000    0.025    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     3195    0.010    0.000    0.015    0.000 surrogate.py:2568(get_fit_params)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:373(__call__)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
     6396    0.004    0.000    0.004    0.000 {built-in method numpy.array}
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:607(_integrate_forward)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      238    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
     2957    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
     3195    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
      130    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
      238    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
       28    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
     3195    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
      322    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1101(coorb_spins_from_copr_spins)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
```

##### NRSur7dq4v2 / geom_dt_0.5_flow_0.01

```text
         27421 function calls in 0.042 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.042    0.042 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.042    0.042 surrogate.py:1722(__call__)
        1    0.000    0.000    0.042    0.042 precessing_surrogate.py:1214(__call__)
        1    0.001    0.001    0.026    0.026 precessing_surrogate.py:1015(__call__)
      128    0.000    0.000    0.025    0.000 precessing_surrogate.py:1061(_eval_comp)
      128    0.006    0.000    0.025    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     3353    0.011    0.000    0.016    0.000 surrogate.py:2568(get_fit_params)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:373(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:583(_one_backward_RK4_step)
     6882    0.004    0.000    0.004    0.000 {built-in method numpy.array}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1108(inertial_waveform_modes)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
     3074    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     3353    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:641(_integrate_backward)
      130    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:607(_integrate_forward)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
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
      750    0.001    0.000    0.001    0.000 surrogate.py:2715(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
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
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         11280 function calls in 0.017 seconds

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
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      750    0.001    0.000    0.001    0.000 surrogate.py:2715(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      750    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:912(_eval_comp)
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
      926    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      204    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
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
      750    0.001    0.000    0.001    0.000 surrogate.py:2715(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      750    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
       16    0.000    0.000    0.000    0.000 precessing_surrogate.py:912(_eval_comp)
      926    0.000    0.000    0.000    0.000 {built-in method numpy.array}
       16    0.000    0.000    0.000    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2135(<genexpr>)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      204    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         11280 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 102 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1722(__call__)
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
      750    0.001    0.000    0.001    0.000 surrogate.py:2715(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      750    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
      926    0.001    0.000    0.001    0.000 {built-in method numpy.array}
       16    0.000    0.000    0.000    0.000 precessing_surrogate.py:912(_eval_comp)
       16    0.000    0.000    0.000    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:508(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      204    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
       22    0.000    0.000    0.000    0.000 surrogate.py:2135(<genexpr>)
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
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:607(_integrate_forward)
      505    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      505    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
      597    0.000    0.000    0.001    0.000 surrogate.py:2715(get_fit_params)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:912(_eval_comp)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      603    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1101(coorb_spins_from_copr_spins)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         12593 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1722(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:1557(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:373(__call__)
      546    0.000    0.000    0.006    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:607(_integrate_forward)
      546    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:297(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:583(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:641(_integrate_backward)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      906    0.001    0.000    0.001    0.000 surrogate.py:2715(get_fit_params)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
        3    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
      493    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:912(_eval_comp)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
     1082    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:508(_initialize)
      268    0.000    0.000    0.000    0.000 precessing_surrogate.py:125(_eval_scalar_fit)
      360    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         5802 function calls in 0.015 seconds

   Ordered by: cumulative time
   List reduced from 88 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.015    0.015 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.015    0.015 surrogate.py:1722(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:1557(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:1214(__call__)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:373(__call__)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:607(_integrate_forward)
      505    0.000    0.000    0.005    0.000 precessing_surrogate.py:273(get_time_deriv_from_index)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:78(rotateWaveform)
      505    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
      597    0.000    0.000    0.001    0.000 surrogate.py:2715(get_fit_params)
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
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:542(_initial_RK4)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1101(coorb_spins_from_copr_spins)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:804(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:29(quatInv)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1125(normalize_spin)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
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
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:1108(inertial_waveform_modes)
      546    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:607(_integrate_forward)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:641(_integrate_backward)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.wignerD_matrices}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:326(_get_t_from_omega)
      906    0.001    0.000    0.001    0.000 surrogate.py:2715(get_fit_params)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:320(get_omega)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:875(__call__)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:912(_eval_comp)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:772(_eval_coorbital_component)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1115(splinterp_many)
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
         9044 function calls in 0.066 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.066    0.066 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.066    0.066 surrogate.py:1722(__call__)
        1    0.003    0.003    0.058    0.058 surrogate.py:933(__call__)
        1    0.001    0.001    0.031    0.031 surrogate.py:741(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        1    0.000    0.000    0.014    0.014 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.014    0.014    0.014    0.014 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.003 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       21    0.004    0.000    0.004    0.000 surrogate.py:2135(<genexpr>)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
        3    0.001    0.000    0.001    0.000 {built-in method numpy.ascontiguousarray}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         9040 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1722(__call__)
        1    0.002    0.002    0.031    0.031 surrogate.py:933(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
       21    0.000    0.000    0.000    0.000 surrogate.py:2135(<genexpr>)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         9044 function calls in 0.104 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.104    0.104 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.103    0.103 surrogate.py:1722(__call__)
        1    0.003    0.003    0.084    0.084 surrogate.py:933(__call__)
        1    0.002    0.002    0.057    0.057 surrogate.py:741(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.015    0.015 {method 'update' of 'dict' objects}
       21    0.015    0.001    0.015    0.001 surrogate.py:2135(<genexpr>)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        9    0.003    0.000    0.003    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        4    0.002    0.001    0.002    0.001 _function_base_impl.py:1413(diff)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
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
        1    0.002    0.002    0.033    0.033 surrogate.py:933(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.008    0.008 surrogate.py:741(_coorbital_to_inertial_frame)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2135(<genexpr>)
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         9017 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1722(__call__)
        1    0.002    0.002    0.034    0.034 surrogate.py:933(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
        1    0.001    0.001    0.009    0.009 surrogate.py:741(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
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
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         9017 function calls in 0.307 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.307    0.307 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.026    0.026    0.307    0.307 surrogate.py:1722(__call__)
        1    0.003    0.003    0.269    0.269 surrogate.py:933(__call__)
        1    0.010    0.010    0.242    0.242 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.100    0.100 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.100    0.100    0.100    0.100 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.073    0.073    0.073    0.073 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.052    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.052    0.026    0.052    0.026 spline_interp_Cwrapper.py:50(interpolate)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        9    0.012    0.001    0.012    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1413(diff)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:731(_search_omega)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         9017 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1722(__call__)
        1    0.002    0.002    0.031    0.031 surrogate.py:933(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.006    0.000    0.006    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:741(_coorbital_to_inertial_frame)
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
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
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
        1    0.002    0.002    0.087    0.087 surrogate.py:933(__call__)
        1    0.002    0.002    0.061    0.061 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.026    0.026 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.026    0.026    0.026    0.026 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
        1    0.021    0.021    0.021    0.021 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         3540 function calls in 0.023 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.023    0.023 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.023    0.023 surrogate.py:1722(__call__)
        1    0.000    0.000    0.020    0.020 surrogate.py:933(__call__)
        1    0.001    0.001    0.015    0.015 surrogate.py:741(_coorbital_to_inertial_frame)
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
       11    0.001    0.000    0.001    0.000 surrogate.py:2135(<genexpr>)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       20    0.000    0.000    0.000    0.000 __init__.py:613(cast)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         3536 function calls in 0.009 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.009    0.009 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.009    0.009 surrogate.py:1722(__call__)
        1    0.000    0.000    0.008    0.008 surrogate.py:933(__call__)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
        1    0.000    0.000    0.003    0.003 surrogate.py:741(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       11    0.000    0.000    0.000    0.000 surrogate.py:2135(<genexpr>)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         3540 function calls in 0.036 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.036    0.036 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.036    0.036 surrogate.py:1722(__call__)
        1    0.000    0.000    0.032    0.032 surrogate.py:933(__call__)
        1    0.001    0.001    0.026    0.026 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.007    0.007 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.007    0.007    0.007    0.007 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.007    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.006    0.003    0.007    0.003 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2135(<genexpr>)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         3536 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1722(__call__)
        1    0.000    0.000    0.009    0.009 surrogate.py:933(__call__)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:741(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       11    0.000    0.000    0.000    0.000 surrogate.py:2135(<genexpr>)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         3523 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1722(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:933(__call__)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
        1    0.000    0.000    0.005    0.005 surrogate.py:741(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         3523 function calls in 0.164 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.164    0.164 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.013    0.013    0.164    0.164 surrogate.py:1722(__call__)
        1    0.000    0.000    0.143    0.143 surrogate.py:933(__call__)
        1    0.006    0.006    0.137    0.137 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.055    0.055    0.055    0.055 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.037    0.037 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.037    0.037    0.037    0.037 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.035    0.017 surrogate.py:91(_splinterp_Cwrapper)
        2    0.034    0.017    0.035    0.017 spline_interp_Cwrapper.py:50(interpolate)
        5    0.008    0.002    0.008    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.006    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.006    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1413(diff)
        2    0.002    0.001    0.002    0.001 surrogate.py:731(_search_omega)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1635(_check_params)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         3523 function calls in 0.008 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.008    0.008 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.008    0.008 surrogate.py:1722(__call__)
        1    0.000    0.000    0.008    0.008 surrogate.py:933(__call__)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:155(__call__)
        1    0.000    0.000    0.003    0.003 surrogate.py:741(_coorbital_to_inertial_frame)
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
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      332    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
       66    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         3523 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.038    0.038 surrogate.py:1722(__call__)
        1    0.000    0.000    0.036    0.036 surrogate.py:933(__call__)
        1    0.001    0.001    0.030    0.030 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.011    0.011    0.011    0.011 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.009    0.009 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.009    0.009 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.007    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      332    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         10972 function calls in 0.060 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.060    0.060 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.060    0.060 surrogate.py:1722(__call__)
        1    0.002    0.002    0.056    0.056 surrogate.py:933(__call__)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.027    0.001 surrogate.py:292(__call__)
        1    0.001    0.001    0.026    0.026 surrogate.py:741(_coorbital_to_inertial_frame)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.003 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       18    0.002    0.000    0.002    0.000 surrogate.py:2135(<genexpr>)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         10968 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.037    0.037 surrogate.py:1722(__call__)
        1    0.002    0.002    0.036    0.036 surrogate.py:933(__call__)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.017    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.007    0.007 surrogate.py:741(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       18    0.000    0.000    0.000    0.000 surrogate.py:2135(<genexpr>)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         10972 function calls in 0.087 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.087    0.087 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.087    0.087 surrogate.py:1722(__call__)
        1    0.002    0.002    0.079    0.079 surrogate.py:933(__call__)
        1    0.002    0.002    0.050    0.050 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.021    0.021 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.020    0.020    0.021    0.021 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
        1    0.016    0.016    0.016    0.016 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.009    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.004    0.009    0.005 spline_interp_Cwrapper.py:50(interpolate)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       18    0.004    0.000    0.004    0.000 surrogate.py:2135(<genexpr>)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         10968 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1722(__call__)
        1    0.002    0.002    0.037    0.037 surrogate.py:933(__call__)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.017    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.008    0.008 surrogate.py:741(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2135(<genexpr>)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         10948 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1722(__call__)
        1    0.002    0.002    0.037    0.037 surrogate.py:933(__call__)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.017    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.008    0.008 surrogate.py:741(_coorbital_to_inertial_frame)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      654    0.000    0.000    0.000    0.000 {built-in method _warnings._filters_mutated_lock_held}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         10948 function calls in 0.290 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.290    0.290 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.021    0.021    0.290    0.290 surrogate.py:1722(__call__)
        1    0.004    0.004    0.259    0.259 surrogate.py:933(__call__)
        1    0.007    0.007    0.228    0.228 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.092    0.092 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.091    0.091    0.092    0.092 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.069    0.069    0.069    0.069 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.052    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.052    0.026    0.052    0.026 spline_interp_Cwrapper.py:50(interpolate)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.027    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        7    0.010    0.001    0.010    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1413(diff)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:731(_search_omega)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5592(append)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         10948 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1722(__call__)
        1    0.002    0.002    0.035    0.035 surrogate.py:933(__call__)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.017    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.009    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:741(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
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
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         10948 function calls in 0.090 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.090    0.090 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.090    0.090 surrogate.py:1722(__call__)
        1    0.002    0.002    0.086    0.086 surrogate.py:933(__call__)
        1    0.002    0.002    0.057    0.057 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.023    0.023    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
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
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

#### PR-90

##### NRSur7dq4 / mks_dt_0.000244140625_flow_0

```text
         9259 function calls in 0.017 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.017    0.017 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.017    0.017 surrogate.py:1722(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:366(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:600(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:868(__call__)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:905(_eval_comp)
      574    0.001    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
      749    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:319(_get_t_from_omega)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:313(get_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         9260 function calls in 0.016 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.016    0.016 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.016    0.016 surrogate.py:1722(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:366(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:600(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:868(__call__)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:905(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
      574    0.001    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
      749    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:319(_get_t_from_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:313(get_omega)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      251    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         9259 function calls in 0.017 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.017    0.017 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.017    0.017 surrogate.py:1722(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:366(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:600(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:868(__call__)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:905(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
      574    0.001    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
      749    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:319(_get_t_from_omega)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:313(get_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         9260 function calls in 0.017 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.017    0.017 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.017    0.017 surrogate.py:1722(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:366(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        1    0.001    0.001    0.002    0.002 precessing_surrogate.py:600(_integrate_forward)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:868(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:905(_eval_comp)
      574    0.001    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
      749    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:319(_get_t_from_omega)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:313(get_omega)
      295    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         4213 function calls in 0.014 seconds

   Ordered by: cumulative time
   List reduced from 85 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.014    0.014 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.014    0.014 surrogate.py:1722(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:1207(__call__)
        4    0.000    0.000    0.006    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.005    0.005 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.005    0.005    0.005    0.005 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:366(__call__)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:600(_integrate_forward)
      238    0.000    0.000    0.002    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:868(__call__)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:905(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
      486    0.001    0.000    0.001    0.000 surrogate.py:2451(get_fit_params)
      238    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      491    0.000    0.000    0.000    0.000 {built-in method numpy.array}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:535(_initial_RK4)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1094(coorb_spins_from_copr_spins)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1118(normalize_spin)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
       13    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:112(<genexpr>)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:1074(rotate_spin)
      108    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         9796 function calls in 0.015 seconds

   Ordered by: cumulative time
   List reduced from 97 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.015    0.015 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.015    0.015 surrogate.py:1722(__call__)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:366(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:868(__call__)
      644    0.002    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:905(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:600(_integrate_forward)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:319(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:313(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
      819    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:118(_eval_scalar_fit)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         4213 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 85 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1722(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:366(__call__)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:600(_integrate_forward)
      238    0.000    0.000    0.002    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:868(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:905(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      486    0.001    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
      238    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        3    0.000    0.000    0.000    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      491    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      248    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:535(_initial_RK4)
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1094(coorb_spins_from_copr_spins)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1118(normalize_spin)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
       13    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:112(<genexpr>)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:1074(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         9796 function calls in 0.014 seconds

   Ordered by: cumulative time
   List reduced from 97 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.014    0.014 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.014    0.014 surrogate.py:1722(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:366(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:868(__call__)
      644    0.001    0.000    0.002    0.000 surrogate.py:2451(get_fit_params)
       42    0.000    0.000    0.002    0.000 precessing_surrogate.py:905(_eval_comp)
       42    0.001    0.000    0.002    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:600(_integrate_forward)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:319(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:313(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      819    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      365    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:118(_eval_scalar_fit)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      650    0.000    0.000    0.000    0.000 __init__.py:271(POINTER)
```

##### NRSur7dq4v2 / mks_dt_0.000244140625_flow_0

```text
         20270 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1722(__call__)
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:1207(__call__)
        1    0.001    0.001    0.017    0.017 precessing_surrogate.py:1008(__call__)
       84    0.000    0.000    0.016    0.000 precessing_surrogate.py:1054(_eval_comp)
       84    0.004    0.000    0.016    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
     2264    0.007    0.000    0.011    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:366(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:600(_integrate_forward)
     4703    0.003    0.000    0.003    0.000 {built-in method numpy.array}
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
     1985    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
     2264    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       86    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:319(_get_t_from_omega)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:313(get_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       18    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
     2264    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      353    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
```

##### NRSur7dq4v2 / mks_dt_0.000244140625_flow_20

```text
         20271 function calls in 0.030 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.030    0.030 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.030    0.030 surrogate.py:1722(__call__)
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1008(__call__)
       84    0.000    0.000    0.017    0.000 precessing_surrogate.py:1054(_eval_comp)
       84    0.004    0.000    0.016    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
     2264    0.008    0.000    0.011    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:366(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:600(_integrate_forward)
     4703    0.003    0.000    0.003    0.000 {built-in method numpy.array}
     1985    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
     2264    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       86    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:319(_get_t_from_omega)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:313(get_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       18    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
     2264    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      326    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      353    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### NRSur7dq4v2 / mks_dt_0.0001220703125_flow_0

```text
         20270 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1722(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1008(__call__)
       84    0.000    0.000    0.017    0.000 precessing_surrogate.py:1054(_eval_comp)
       84    0.004    0.000    0.017    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
     2264    0.008    0.000    0.011    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:366(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:600(_integrate_forward)
     4703    0.003    0.000    0.003    0.000 {built-in method numpy.array}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     1985    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
     2264    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       86    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:319(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:313(get_omega)
       18    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
     2264    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      353    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
```

##### NRSur7dq4v2 / mks_dt_0.0001220703125_flow_20

```text
         20271 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1722(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1008(__call__)
       84    0.000    0.000    0.016    0.000 precessing_surrogate.py:1054(_eval_comp)
       84    0.004    0.000    0.016    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
     2264    0.007    0.000    0.011    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:366(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:600(_integrate_forward)
     4703    0.003    0.000    0.003    0.000 {built-in method numpy.array}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     1985    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1101(inertial_waveform_modes)
     2264    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      279    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
       86    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:319(_get_t_from_omega)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:313(get_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
     2264    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
       18    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      353    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
```

##### NRSur7dq4v2 / geom_dt_0.1_flow_0

```text
         21674 function calls in 0.043 seconds

   Ordered by: cumulative time
   List reduced from 85 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.043    0.043 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.043    0.043 surrogate.py:1722(__call__)
        1    0.000    0.000    0.043    0.043 precessing_surrogate.py:1207(__call__)
        1    0.001    0.001    0.026    0.026 precessing_surrogate.py:1008(__call__)
      128    0.000    0.000    0.025    0.000 precessing_surrogate.py:1054(_eval_comp)
      128    0.006    0.000    0.025    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
     3195    0.010    0.000    0.015    0.000 surrogate.py:2564(get_fit_params)
        4    0.000    0.000    0.009    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.008    0.008 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.008    0.008 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:366(__call__)
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:600(_integrate_forward)
     6395    0.004    0.000    0.004    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
      238    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
     2957    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     3195    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
      130    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
      238    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
       28    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
     3195    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:535(_initial_RK4)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      318    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1094(coorb_spins_from_copr_spins)
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1118(normalize_spin)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
       96    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:1074(rotate_spin)
```

##### NRSur7dq4v2 / geom_dt_0.1_flow_0.01

```text
         27415 function calls in 0.043 seconds

   Ordered by: cumulative time
   List reduced from 97 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.043    0.043 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.043    0.043 surrogate.py:1722(__call__)
        1    0.000    0.000    0.043    0.043 precessing_surrogate.py:1207(__call__)
        1    0.001    0.001    0.027    0.027 precessing_surrogate.py:1008(__call__)
      128    0.000    0.000    0.025    0.000 precessing_surrogate.py:1054(_eval_comp)
      128    0.006    0.000    0.025    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
     3353    0.011    0.000    0.016    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:366(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:576(_one_backward_RK4_step)
     6881    0.004    0.000    0.004    0.000 {built-in method numpy.array}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1101(inertial_waveform_modes)
     3074    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     3353    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:634(_integrate_backward)
      130    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:600(_integrate_forward)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:319(_get_t_from_omega)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:313(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:501(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
       28    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
     3353    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      461    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:118(_eval_scalar_fit)
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4v2 / geom_dt_0.5_flow_0

```text
         21674 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 85 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.037    0.037 surrogate.py:1722(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:1207(__call__)
        1    0.001    0.001    0.026    0.026 precessing_surrogate.py:1008(__call__)
      128    0.000    0.000    0.025    0.000 precessing_surrogate.py:1054(_eval_comp)
      128    0.007    0.000    0.025    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
     3195    0.010    0.000    0.015    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:366(__call__)
     6395    0.004    0.000    0.004    0.000 {built-in method numpy.array}
        1    0.001    0.001    0.004    0.004 precessing_surrogate.py:600(_integrate_forward)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      238    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
     2957    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     3195    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
      130    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
      238    0.001    0.000    0.001    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
       28    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
     3195    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:535(_initial_RK4)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      318    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      229    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1094(coorb_spins_from_copr_spins)
       96    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1118(normalize_spin)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
       13    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:114(<genexpr>)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
```

##### NRSur7dq4v2 / geom_dt_0.5_flow_0.01

```text
         27415 function calls in 0.042 seconds

   Ordered by: cumulative time
   List reduced from 97 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.042    0.042 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.042    0.042 surrogate.py:1722(__call__)
        1    0.000    0.000    0.042    0.042 precessing_surrogate.py:1207(__call__)
        1    0.001    0.001    0.026    0.026 precessing_surrogate.py:1008(__call__)
      128    0.000    0.000    0.025    0.000 precessing_surrogate.py:1054(_eval_comp)
      128    0.006    0.000    0.025    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
     3353    0.011    0.000    0.016    0.000 surrogate.py:2564(get_fit_params)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:366(__call__)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:576(_one_backward_RK4_step)
     6881    0.004    0.000    0.004    0.000 {built-in method numpy.array}
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      279    0.000    0.000    0.003    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1101(inertial_waveform_modes)
     3074    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
     3353    0.001    0.000    0.002    0.000 _function_base_impl.py:935(copy)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:634(_integrate_backward)
      130    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:600(_integrate_forward)
      279    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:319(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:313(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
       28    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
     3353    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      461    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      396    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      117    0.000    0.000    0.000    0.000 precessing_surrogate.py:118(_eval_scalar_fit)
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         11274 function calls in 0.017 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.017    0.017 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.017    0.017 surrogate.py:1722(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1550(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:366(__call__)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:600(_integrate_forward)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
      546    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
      750    0.001    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:868(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:319(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      750    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:313(get_omega)
      925    0.001    0.000    0.001    0.000 {built-in method numpy.array}
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:905(_eval_comp)
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      204    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:118(_eval_scalar_fit)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         11275 function calls in 0.017 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.017    0.017 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.017    0.017 surrogate.py:1722(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:1550(__call__)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:366(__call__)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:600(_integrate_forward)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      546    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      750    0.001    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:868(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:319(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:313(get_omega)
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:905(_eval_comp)
      750    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
      925    0.000    0.000    0.000    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      204    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:118(_eval_scalar_fit)
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         11274 function calls in 0.018 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.018    0.018 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.018    0.018 surrogate.py:1722(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:1550(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:366(__call__)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:600(_integrate_forward)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      546    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
      750    0.001    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:868(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:319(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:313(get_omega)
      750    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:905(_eval_comp)
      925    0.001    0.000    0.001    0.000 {built-in method numpy.array}
       16    0.000    0.000    0.000    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      204    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:118(_eval_scalar_fit)
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         11275 function calls in 0.017 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.017    0.017 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.017    0.017 surrogate.py:1722(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1550(__call__)
        1    0.000    0.000    0.017    0.017 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:366(__call__)
        1    0.001    0.001    0.005    0.005 precessing_surrogate.py:600(_integrate_forward)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      546    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:78(rotateWaveform)
        1    0.002    0.002    0.002    0.002 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:634(_integrate_backward)
      750    0.001    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:868(__call__)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:319(_get_t_from_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:313(get_omega)
      750    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
      925    0.001    0.000    0.001    0.000 {built-in method numpy.array}
       16    0.000    0.000    0.001    0.000 precessing_surrogate.py:905(_eval_comp)
       16    0.000    0.000    0.000    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
      646    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      204    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      129    0.000    0.000    0.000    0.000 precessing_surrogate.py:118(_eval_scalar_fit)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         5796 function calls in 0.021 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.021    0.021 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.021    0.021 surrogate.py:1722(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:1550(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:1207(__call__)
        4    0.000    0.000    0.010    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.009    0.009 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.009    0.009 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:366(__call__)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:600(_integrate_forward)
      505    0.000    0.000    0.005    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
      505    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
      597    0.000    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:868(__call__)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:905(_eval_comp)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      602    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:535(_initial_RK4)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1094(coorb_spins_from_copr_spins)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1118(normalize_spin)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       13    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:112(<genexpr>)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         12587 function calls in 0.020 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.020    0.020 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.020    0.020 surrogate.py:1722(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:1550(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:366(__call__)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
      546    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:600(_integrate_forward)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:634(_integrate_backward)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      906    0.001    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:319(_get_t_from_omega)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:313(get_omega)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:868(__call__)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:905(_eval_comp)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     1081    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
      268    0.000    0.000    0.000    0.000 precessing_surrogate.py:118(_eval_scalar_fit)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      360    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         5796 function calls in 0.014 seconds

   Ordered by: cumulative time
   List reduced from 87 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.014    0.014 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.014    0.014 surrogate.py:1722(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:1550(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:366(__call__)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:600(_integrate_forward)
      505    0.000    0.000    0.005    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
      505    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      597    0.000    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:868(__call__)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:905(_eval_comp)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:935(copy)
      602    0.000    0.000    0.000    0.000 {built-in method numpy.array}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:535(_initial_RK4)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
       22    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
       92    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:797(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:1094(coorb_spins_from_copr_spins)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:1118(normalize_spin)
        1    0.000    0.000    0.000    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.000    0.000 _linalg.py:2598(norm)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:1074(rotate_spin)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:931(_copy_dispatcher)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         12587 function calls in 0.019 seconds

   Ordered by: cumulative time
   List reduced from 99 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.019    0.019 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.019    0.019 surrogate.py:1722(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:1550(__call__)
        1    0.000    0.000    0.019    0.019 precessing_surrogate.py:1207(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:366(__call__)
      546    0.000    0.000    0.005    0.000 precessing_surrogate.py:266(get_time_deriv_from_index)
       13    0.000    0.000    0.005    0.000 precessing_surrogate.py:290(get_time_deriv)
        3    0.000    0.000    0.004    0.001 precessing_surrogate.py:576(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      546    0.004    0.000    0.004    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit_batch_dydt}
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:600(_integrate_forward)
        1    0.001    0.001    0.003    0.003 precessing_surrogate.py:634(_integrate_backward)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:1101(inertial_waveform_modes)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:78(rotateWaveform)
        1    0.003    0.003    0.003    0.003 {built-in method gwsurrogate.precessing_utils._utils.rotate_waveform}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      906    0.001    0.000    0.001    0.000 surrogate.py:2711(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:319(_get_t_from_omega)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:313(get_omega)
      584    0.000    0.000    0.001    0.000 _internal.py:279(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:868(__call__)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:935(copy)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:905(_eval_comp)
       20    0.000    0.000    0.001    0.000 precessing_surrogate.py:765(_eval_coorbital_component)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:1108(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     1081    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:501(_initialize)
      268    0.000    0.000    0.000    0.000 precessing_surrogate.py:118(_eval_scalar_fit)
      668    0.000    0.000    0.000    0.000 _internal.py:262(__init__)
      360    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      438    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      497    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         9044 function calls in 0.063 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.063    0.063 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.063    0.063 surrogate.py:1722(__call__)
        1    0.004    0.004    0.058    0.058 surrogate.py:933(__call__)
        1    0.001    0.001    0.031    0.031 surrogate.py:741(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
        1    0.000    0.000    0.014    0.014 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.014    0.014    0.014    0.014 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.005    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 {method 'update' of 'dict' objects}
       21    0.003    0.000    0.003    0.000 surrogate.py:2131(<genexpr>)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
        9    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
        3    0.001    0.000    0.001    0.000 {built-in method numpy.ascontiguousarray}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         9040 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1722(__call__)
        1    0.002    0.002    0.032    0.032 surrogate.py:933(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.021    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.007    0.007 surrogate.py:741(_coorbital_to_inertial_frame)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
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
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       21    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         9044 function calls in 0.104 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.104    0.104 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.104    0.104 surrogate.py:1722(__call__)
        1    0.003    0.003    0.085    0.085 surrogate.py:933(__call__)
        1    0.002    0.002    0.059    0.059 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.024    0.024 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.023    0.023    0.024    0.024 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
        1    0.018    0.018    0.018    0.018 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.015    0.015 {method 'update' of 'dict' objects}
       21    0.015    0.001    0.015    0.001 surrogate.py:2131(<genexpr>)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.008    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        4    0.003    0.001    0.003    0.001 _function_base_impl.py:1413(diff)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         9040 function calls in 0.033 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.033    0.033 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.033    0.033 surrogate.py:1722(__call__)
        1    0.002    0.002    0.032    0.032 surrogate.py:933(__call__)
       12    0.000    0.000    0.021    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.021    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.021    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.014    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.008    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.008    0.008 surrogate.py:741(_coorbital_to_inertial_frame)
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
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
       21    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
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
        1    0.000    0.000    0.034    0.034 surrogate.py:1722(__call__)
        1    0.002    0.002    0.033    0.033 surrogate.py:933(__call__)
       12    0.000    0.000    0.021    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.021    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.021    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.014    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
        1    0.001    0.001    0.009    0.009 surrogate.py:741(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.008    0.000 nodeFunction.py:111(__call__)
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
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         9017 function calls in 0.303 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.303    0.303 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.025    0.025    0.303    0.303 surrogate.py:1722(__call__)
        1    0.003    0.003    0.266    0.266 surrogate.py:933(__call__)
        1    0.009    0.009    0.240    0.240 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.100    0.100 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.100    0.100    0.100    0.100 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.073    0.073    0.073    0.073 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.051    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.051    0.026    0.051    0.026 spline_interp_Cwrapper.py:50(interpolate)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.022    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.014    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
        9    0.011    0.001    0.011    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.008    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1413(diff)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:731(_search_omega)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         9017 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1722(__call__)
        1    0.002    0.002    0.030    0.030 surrogate.py:933(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.022    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.021    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      177    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      177    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.008    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.006    0.000    0.006    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:741(_coorbital_to_inertial_frame)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         9017 function calls in 0.090 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.090    0.090 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.090    0.090 surrogate.py:1722(__call__)
        1    0.002    0.002    0.085    0.085 surrogate.py:933(__call__)
        1    0.002    0.002    0.060    0.060 surrogate.py:741(_coorbital_to_inertial_frame)
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
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
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
        1    0.000    0.000    0.020    0.020 surrogate.py:933(__call__)
        1    0.001    0.001    0.015    0.015 surrogate.py:741(_coorbital_to_inertial_frame)
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
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         3536 function calls in 0.009 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.009    0.009 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.009    0.009 surrogate.py:1722(__call__)
        1    0.000    0.000    0.008    0.008 surrogate.py:933(__call__)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:155(__call__)
        1    0.000    0.000    0.003    0.003 surrogate.py:741(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       11    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         3540 function calls in 0.036 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.036    0.036 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.036    0.036 surrogate.py:1722(__call__)
        1    0.000    0.000    0.032    0.032 surrogate.py:933(__call__)
        1    0.001    0.001    0.026    0.026 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.009    0.009    0.009    0.009 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.007    0.007 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.007    0.007    0.007    0.007 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.007    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.006    0.003    0.007    0.003 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2131(<genexpr>)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1635(_check_params)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         3536 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1722(__call__)
        1    0.000    0.000    0.009    0.009 surrogate.py:933(__call__)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:741(_coorbital_to_inertial_frame)
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
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       11    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         3523 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.010    0.010 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.010    0.010 surrogate.py:1722(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:933(__call__)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
        1    0.000    0.000    0.004    0.004 surrogate.py:741(_coorbital_to_inertial_frame)
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
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      332    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 surrogate.py:1635(_check_params)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         3523 function calls in 0.165 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.165    0.165 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.014    0.014    0.165    0.165 surrogate.py:1722(__call__)
        1    0.000    0.000    0.144    0.144 surrogate.py:933(__call__)
        1    0.006    0.006    0.138    0.138 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.055    0.055    0.055    0.055 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.038    0.038 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.038    0.038    0.038    0.038 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.035    0.018 surrogate.py:91(_splinterp_Cwrapper)
        2    0.035    0.018    0.035    0.018 spline_interp_Cwrapper.py:50(interpolate)
        5    0.007    0.001    0.007    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.002    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1413(diff)
        2    0.001    0.001    0.001    0.001 surrogate.py:731(_search_omega)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1635(_check_params)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         3523 function calls in 0.008 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.008    0.008 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.008    0.008 surrogate.py:1722(__call__)
        1    0.000    0.000    0.008    0.008 surrogate.py:933(__call__)
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.004    0.000 nodeFunction.py:220(__call__)
       66    0.001    0.000    0.004    0.000 nodeFunction.py:155(__call__)
        1    0.000    0.000    0.003    0.003 surrogate.py:741(_coorbital_to_inertial_frame)
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
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      332    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
        1    0.000    0.000    0.000    0.000 surrogate.py:1635(_check_params)
       66    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         3523 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.038    0.038 surrogate.py:1722(__call__)
        1    0.000    0.000    0.035    0.035 surrogate.py:933(__call__)
        1    0.001    0.001    0.030    0.030 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.011    0.011    0.011    0.011 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.009    0.009 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.008    0.008    0.009    0.009 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.008    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
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
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1242(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:1635(_check_params)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         10972 function calls in 0.059 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.059    0.059 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.001    0.001    0.059    0.059 surrogate.py:1722(__call__)
        1    0.002    0.002    0.056    0.056 surrogate.py:933(__call__)
        1    0.001    0.001    0.027    0.027 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
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
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         10968 function calls in 0.036 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.036    0.036 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.036    0.036 surrogate.py:1722(__call__)
        1    0.002    0.002    0.035    0.035 surrogate.py:933(__call__)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:741(_coorbital_to_inertial_frame)
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
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
       18    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         10972 function calls in 0.097 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.096    0.096 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.096    0.096 surrogate.py:1722(__call__)
        1    0.003    0.003    0.081    0.081 surrogate.py:933(__call__)
        1    0.002    0.002    0.051    0.051 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.022    0.022 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.022    0.022 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.012    0.012 {method 'update' of 'dict' objects}
       18    0.012    0.001    0.012    0.001 surrogate.py:2131(<genexpr>)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.009    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.009    0.005 spline_interp_Cwrapper.py:50(interpolate)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         10968 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1722(__call__)
        1    0.002    0.002    0.037    0.037 surrogate.py:933(__call__)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.017    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.007    0.007 surrogate.py:741(_coorbital_to_inertial_frame)
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
       18    0.000    0.000    0.000    0.000 surrogate.py:2131(<genexpr>)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         10948 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1722(__call__)
        1    0.002    0.002    0.037    0.037 surrogate.py:933(__call__)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.001    0.001    0.008    0.008 surrogate.py:741(_coorbital_to_inertial_frame)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.002    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5592(append)
      654    0.000    0.000    0.000    0.000 {built-in method _warnings._acquire_lock}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         10948 function calls in 0.293 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.293    0.293 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.017    0.017    0.293    0.293 surrogate.py:1722(__call__)
        1    0.002    0.002    0.266    0.266 surrogate.py:933(__call__)
        1    0.008    0.008    0.229    0.229 surrogate.py:741(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.093    0.093 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.092    0.092    0.093    0.093 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.070    0.070    0.070    0.070 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.051    0.026 surrogate.py:91(_splinterp_Cwrapper)
        2    0.051    0.025    0.051    0.026 spline_interp_Cwrapper.py:50(interpolate)
       11    0.000    0.000    0.034    0.003 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.034    0.003 surrogate.py:417(__call__)
       19    0.000    0.000    0.033    0.002 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
       23    0.015    0.001    0.015    0.001 {method 'dot' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
        7    0.010    0.001    0.010    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1413(diff)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        2    0.002    0.001    0.002    0.001 surrogate.py:731(_search_omega)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        2    0.001    0.000    0.001    0.000 _function_base_impl.py:5592(append)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         10948 function calls in 0.036 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.036    0.036 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.000    0.000    0.036    0.036 surrogate.py:1722(__call__)
        1    0.002    0.002    0.035    0.035 surrogate.py:933(__call__)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.011    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.005    0.005 surrogate.py:741(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 surrogate.py:731(_search_omega)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
      654    0.000    0.000    0.000    0.000 {built-in method _warnings._acquire_lock}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         10948 function calls in 0.090 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.090    0.090 benchmark_surrogate_evaluations.py:355(evaluate_case)
        1    0.002    0.002    0.090    0.090 surrogate.py:1722(__call__)
        1    0.002    0.002    0.086    0.086 surrogate.py:933(__call__)
        1    0.002    0.002    0.057    0.057 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.026    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.026    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.023    0.023    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.019    0.019    0.019    0.019 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.011    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.011    0.005 spline_interp_Cwrapper.py:50(interpolate)
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.010    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      218    0.001    0.000    0.002    0.000 _py_warnings.py:320(_add_filter)
        7    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1242(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
      218    0.000    0.000    0.001    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1413(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1234(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
        1    0.000    0.000    0.000    0.000 surrogate.py:917(_TaylorT3_phase_22)
```
