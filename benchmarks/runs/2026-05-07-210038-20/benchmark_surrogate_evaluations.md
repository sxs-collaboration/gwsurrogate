# GWSurrogate Evaluation Timing

Generated: 2026-05-07T21:00:33.652932+00:00

Times below are seconds per model evaluation. Raw repeats and context are in the JSON output.

PNG timing table: `test/benchmark_surrogate_evaluations.png`

## Summary

### master

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0200258` s, median `0.0203382` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0201028` s, median `0.0202421` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0234085` s, median `0.0241182` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0232339` s, median `0.0232737` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0202006` s, median `0.0203152` s
- `dt=0.1 M`, `f_low=0.01`: best `0.019458` s, median `0.01958` s
- `dt=0.5 M`, `f_low=0`: best `0.0155247` s, median `0.0156718` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0186845` s, median `0.0188159` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0330991` s, median `0.0333157` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0328706` s, median `0.0330064` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0342992` s, median `0.0344675` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0329098` s, median `0.0331527` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0369378` s, median `0.0372823` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0345915` s, median `0.0348399` s
- `dt=0.5 M`, `f_low=0`: best `0.0292824` s, median `0.0298608` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0326244` s, median `0.0332915` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.079969` s, median `0.0802609` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0469974` s, median `0.0470782` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.120529` s, median `0.120658` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0481109` s, median `0.0482977` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0488158` s, median `0.0491453` s
- `dt=0.1 M`, `f_low=0.002`: best `0.322683` s, median `0.322861` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0458343` s, median `0.0462223` s
- `dt=0.5 M`, `f_low=0.002`: best `0.102898` s, median `0.103253` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0280095` s, median `0.0282667` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0133959` s, median `0.0136164` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0428231` s, median `0.0428989` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0144328` s, median `0.0145319` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.014555` s, median `0.0148172` s
- `dt=0.1 M`, `f_low=0.002`: best `0.178386` s, median `0.178668` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0126152` s, median `0.0126871` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0450719` s, median `0.0451147` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0769479` s, median `0.0771454` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0544162` s, median `0.0544874` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.103506` s, median `0.103645` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0564057` s, median `0.0564869` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0567122` s, median `0.056874` s
- `dt=0.1 M`, `f_low=0.002`: best `0.304081` s, median `0.30473` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0533877` s, median `0.0539119` s
- `dt=0.5 M`, `f_low=0.002`: best `0.107228` s, median `0.107926` s

### PR-72

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0212204` s, median `0.0212615` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0209611` s, median `0.0214252` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0245444` s, median `0.025082` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0241814` s, median `0.0241946` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0201381` s, median `0.0202958` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0195461` s, median `0.0198834` s
- `dt=0.5 M`, `f_low=0`: best `0.0154136` s, median `0.0155866` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0184884` s, median `0.018657` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0323355` s, median `0.0324991` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0316832` s, median `0.0320226` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.033524` s, median `0.0336076` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.032539` s, median `0.032587` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0362236` s, median `0.0362614` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0334611` s, median `0.0336506` s
- `dt=0.5 M`, `f_low=0`: best `0.0286904` s, median `0.0288291` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0320744` s, median `0.0323427` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0576909` s, median `0.0581141` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0251759` s, median `0.0255361` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0997944` s, median `0.100033` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0266` s, median `0.026731` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0269377` s, median `0.0271934` s
- `dt=0.1 M`, `f_low=0.002`: best `0.301468` s, median `0.301956` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0239575` s, median `0.0240189` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0822673` s, median `0.0824947` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.021402` s, median `0.0214925` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.00670953` s, median `0.00675199` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0359519` s, median `0.0360538` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.00757509` s, median `0.00761633` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.00801839` s, median `0.00807365` s
- `dt=0.1 M`, `f_low=0.002`: best `0.172221` s, median `0.172571` s
- `dt=0.5 M`, `f_low=0.01`: best `0.00586297` s, median `0.00590152` s
- `dt=0.5 M`, `f_low=0.002`: best `0.037891` s, median `0.0379474` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0503642` s, median `0.0506247` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0282026` s, median `0.0282948` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0776086` s, median `0.0776698` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0295775` s, median `0.0296575` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0300283` s, median `0.0300767` s
- `dt=0.1 M`, `f_low=0.002`: best `0.282268` s, median `0.282416` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0272037` s, median `0.027585` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0816861` s, median `0.0820633` s

## Context

### master

- Git branch: `master`
- Git commit: `d106ade3c392e29e3971e17dec1a101b8346c61d`
- Git describe: `v1.1.8-26-gd106ade`
- Python: `3.14.4 (main, Apr  8 2026, 02:27:22) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1010-azure x86_64`
- CPU count: `4`
- Conda env: `unknown`

Submodules:

- `gwsurrogate/eval_pysur`: `671476d6e5e911720bd8ccc250098d848b824ddb` initialized ((671476d))

### PR-72

- Git branch: `unknown`
- Git commit: `d265ec423ec7d5f12f918d429fd6d61257b4295b`
- Git describe: `v1.1.8-25-gd265ec4`
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
Model name:                              AMD EPYC 9V74 80-Core Processor
CPU family:                              25
Model:                                   17
Thread(s) per core:                      2
Core(s) per socket:                      2
Socket(s):                               1
Stepping:                                1
BogoMIPS:                                5192.28
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

#### PR-72

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
BogoMIPS:                                5192.28
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
         27100 function calls in 0.029 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.029    0.029 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2806    0.001    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
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
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         27101 function calls in 0.029 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.029    0.029 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.007    0.007 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2806    0.001    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
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
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
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
         27100 function calls in 0.033 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.033    0.033 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.033    0.033 surrogate.py:1721(__call__)
        1    0.000    0.000    0.032    0.032 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:110(rotateWaveform)
        1    0.006    0.006    0.008    0.008 precessing_surrogate.py:42(_wignerD_matrices)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
     1633    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         27101 function calls in 0.032 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.032    0.032 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.032    0.032 surrogate.py:1721(__call__)
        1    0.000    0.000    0.032    0.032 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:110(rotateWaveform)
        1    0.006    0.006    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
     2806    0.001    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
     1633    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
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
        4    0.000    0.000    0.006    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.006    0.006 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.006    0.006    0.006    0.006 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.004    0.004    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2390    0.001    0.000    0.004    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
     2390    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      486    0.001    0.000    0.001    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     1252    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         27707 function calls in 0.029 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.029    0.029 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2876    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1703    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         20326 function calls in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 91 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.022    0.022 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.022    0.022 surrogate.py:1721(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      238    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.004    0.004    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2390    0.001    0.000    0.004    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
     2390    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
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
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         27707 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.004    0.004    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2876    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1703    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
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
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         43097 function calls in 0.045 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.045    0.045 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.045    0.045 surrogate.py:1721(__call__)
        1    0.000    0.000    0.045    0.045 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.045    0.045 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:626(_integrate_forward)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         43096 function calls in 0.047 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:626(_integrate_forward)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:660(_integrate_backward)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
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
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

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
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.001    0.001    0.016    0.016 precessing_surrogate.py:626(_integrate_forward)
     5135    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         35075 function calls in 0.048 seconds

   Ordered by: cumulative time
   List reduced from 94 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.048    0.048 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.048    0.048 surrogate.py:1721(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.048    0.048 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.020    0.020 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.018    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
     4637    0.003    0.000    0.012    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
        4    0.000    0.000    0.011    0.003 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
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
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
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
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5274    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:660(_integrate_backward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:626(_integrate_forward)
     5274    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:345(_get_t_from_omega)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:339(get_omega)
     2744    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
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
        1    0.000    0.000    0.040    0.040 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.020    0.020 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.018    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.003    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     4637    0.003    0.000    0.012    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
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
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         44452 function calls in 0.047 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.016    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5274    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.014    0.014 precessing_surrogate.py:110(rotateWaveform)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
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
     2744    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.001    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         76481 function calls in 0.117 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.118    0.118 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.118    0.118 surrogate.py:1721(__call__)
        1    0.003    0.003    0.113    0.113 surrogate.py:934(__call__)
       12    0.000    0.000    0.077    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.077    0.006 surrogate.py:417(__call__)
       22    0.000    0.000    0.076    0.003 surrogate.py:292(__call__)
      177    0.000    0.000    0.068    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.068    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.062    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.062    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.062    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.039    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.037    0.000 _gpr.py:373(predict)
        1    0.002    0.002    0.032    0.032 surrogate.py:742(_coorbital_to_inertial_frame)
      354    0.004    0.000    0.032    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.015    0.000 kernels.py:833(__call__)
        1    0.000    0.000    0.014    0.014 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.013    0.013    0.014    0.014 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.011    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
        2    0.000    0.000    0.005    0.003 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
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
        1    0.000    0.000    0.003    0.003 {method 'update' of 'dict' objects}
       21    0.003    0.000    0.003    0.000 surrogate.py:2126(<genexpr>)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         76477 function calls in 0.086 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.086    0.086 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.086    0.086 surrogate.py:1721(__call__)
        1    0.002    0.002    0.085    0.085 surrogate.py:934(__call__)
       12    0.000    0.000    0.075    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.075    0.006 surrogate.py:417(__call__)
       22    0.000    0.000    0.075    0.003 surrogate.py:292(__call__)
      177    0.000    0.000    0.068    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.068    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.062    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.062    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.062    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.038    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.036    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.031    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.015    0.000 kernels.py:833(__call__)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
       26    0.006    0.000    0.006    0.000 {method 'dot' of 'numpy.ndarray' objects}
     6905    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      531    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      354    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      354    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
     1416    0.001    0.000    0.003    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         76481 function calls in 0.156 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.156    0.156 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.156    0.156 surrogate.py:1721(__call__)
        1    0.003    0.003    0.136    0.136 surrogate.py:934(__call__)
       12    0.000    0.000    0.076    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.076    0.006 surrogate.py:417(__call__)
       22    0.000    0.000    0.076    0.003 surrogate.py:292(__call__)
      177    0.000    0.000    0.068    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.068    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.062    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.062    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.062    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.003    0.003    0.056    0.056 surrogate.py:742(_coorbital_to_inertial_frame)
      354    0.002    0.000    0.039    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.037    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.031    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.016    0.016 {method 'update' of 'dict' objects}
       21    0.016    0.001    0.016    0.001 surrogate.py:2126(<genexpr>)
      177    0.001    0.000    0.015    0.000 kernels.py:833(__call__)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.011    0.000 kernels.py:931(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
     1062    0.003    0.000    0.009    0.000 validation.py:371(_num_samples)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
     6905    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      531    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         76477 function calls in 0.086 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.087    0.087 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.087    0.087 surrogate.py:1721(__call__)
        1    0.002    0.002    0.086    0.086 surrogate.py:934(__call__)
       12    0.000    0.000    0.075    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.075    0.006 surrogate.py:417(__call__)
       22    0.000    0.000    0.074    0.003 surrogate.py:292(__call__)
      177    0.000    0.000    0.067    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.067    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.062    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.062    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.061    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.038    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.036    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.031    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.015    0.000 kernels.py:833(__call__)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.011    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.002    0.000    0.007    0.000 kernels.py:1525(__call__)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
     6905    0.002    0.000    0.005    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      531    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      354    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      354    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
     1416    0.001    0.000    0.003    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         76454 function calls in 0.087 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.088    0.088 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.088    0.088 surrogate.py:1721(__call__)
        1    0.002    0.002    0.087    0.087 surrogate.py:934(__call__)
       12    0.000    0.000    0.075    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.075    0.006 surrogate.py:417(__call__)
       22    0.000    0.000    0.075    0.003 surrogate.py:292(__call__)
      177    0.000    0.000    0.068    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.068    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.062    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.062    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.062    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.038    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.036    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.031    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.015    0.000 kernels.py:833(__call__)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.011    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
       26    0.006    0.000    0.006    0.000 {method 'dot' of 'numpy.ndarray' objects}
     6905    0.002    0.000    0.004    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      531    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      354    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      354    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
     1416    0.001    0.000    0.003    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         76454 function calls in 0.358 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.359    0.359 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.027    0.027    0.359    0.359 surrogate.py:1721(__call__)
        1    0.003    0.003    0.320    0.320 surrogate.py:934(__call__)
        1    0.009    0.009    0.241    0.241 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.101    0.101 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.101    0.101    0.101    0.101 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.074    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.074    0.006 surrogate.py:417(__call__)
        1    0.074    0.074    0.074    0.074 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       22    0.000    0.000    0.074    0.003 surrogate.py:292(__call__)
      177    0.000    0.000    0.067    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.067    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.062    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.061    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.061    0.000 evaluate_fit.py:128(GPR_predict)
        2    0.000    0.000    0.048    0.024 surrogate.py:91(_splinterp_Cwrapper)
        2    0.047    0.024    0.048    0.024 spline_interp_Cwrapper.py:50(interpolate)
      354    0.002    0.000    0.038    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.036    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.031    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.023    0.000 _base.py:297(predict)
      177    0.001    0.000    0.023    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.015    0.000 kernels.py:833(__call__)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
        9    0.012    0.001    0.012    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.011    0.000 kernels.py:931(__call__)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.002    0.000    0.007    0.000 kernels.py:1525(__call__)
        4    0.005    0.001    0.005    0.001 _function_base_impl.py:1402(diff)
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
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         76454 function calls in 0.084 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.085    0.085 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.085    0.085 surrogate.py:1721(__call__)
        1    0.002    0.002    0.084    0.084 surrogate.py:934(__call__)
       12    0.000    0.000    0.075    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.075    0.006 surrogate.py:417(__call__)
       22    0.000    0.000    0.075    0.003 surrogate.py:292(__call__)
      177    0.000    0.000    0.068    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.068    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.062    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.062    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.062    0.000 evaluate_fit.py:128(GPR_predict)
      354    0.002    0.000    0.038    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.037    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.031    0.000 validation.py:725(check_array)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
      177    0.001    0.000    0.015    0.000 kernels.py:833(__call__)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.011    0.000 kernels.py:931(__call__)
     1062    0.003    0.000    0.009    0.000 validation.py:371(_num_samples)
      177    0.003    0.000    0.007    0.000 kernels.py:1525(__call__)
       26    0.006    0.000    0.006    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
     6905    0.002    0.000    0.004    0.000 {built-in method builtins.isinstance}
      177    0.000    0.000    0.004    0.000 kernels.py:1239(__call__)
      531    0.000    0.000    0.004    0.000 _tags.py:250(get_tags)
      354    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      531    0.001    0.000    0.003    0.000 _py_warnings.py:320(_add_filter)
      177    0.000    0.000    0.003    0.000 kernels.py:1369(__call__)
      177    0.000    0.000    0.003    0.000 validation.py:1621(check_is_fitted)
      531    0.000    0.000    0.003    0.000 base.py:1166(__sklearn_tags__)
      708    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
    10443    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      354    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
     1062    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      531    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      354    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      177    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         76454 function calls in 0.142 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.143    0.143 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.143    0.143 surrogate.py:1721(__call__)
        1    0.002    0.002    0.138    0.138 surrogate.py:934(__call__)
       12    0.000    0.000    0.076    0.006 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.076    0.006 surrogate.py:417(__call__)
       22    0.000    0.000    0.076    0.003 surrogate.py:292(__call__)
      177    0.000    0.000    0.069    0.000 nodeFunction.py:205(__call__)
      177    0.002    0.000    0.068    0.000 nodeFunction.py:110(__call__)
      177    0.000    0.000    0.063    0.000 nodeFunction.py:96(__call__)
      177    0.000    0.000    0.063    0.000 evaluate_fit.py:247(gprfitEvaluator)
      177    0.001    0.000    0.062    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.002    0.002    0.058    0.058 surrogate.py:742(_coorbital_to_inertial_frame)
      354    0.002    0.000    0.039    0.000 validation.py:2793(validate_data)
      177    0.002    0.000    0.037    0.000 _gpr.py:373(predict)
      354    0.004    0.000    0.032    0.000 validation.py:725(check_array)
        1    0.000    0.000    0.025    0.025 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.025    0.025    0.025    0.025 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.024    0.000 _base.py:297(predict)
      177    0.001    0.000    0.024    0.000 _base.py:287(_decision_function)
        1    0.020    0.020    0.020    0.020 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.001    0.000    0.015    0.000 kernels.py:833(__call__)
      354    0.003    0.000    0.012    0.000 validation.py:103(_assert_all_finite)
      177    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
     1062    0.004    0.000    0.009    0.000 validation.py:371(_num_samples)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
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
         28686 function calls in 0.040 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.040    0.040 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.040    0.040 surrogate.py:1721(__call__)
        1    0.000    0.000    0.038    0.038 surrogate.py:934(__call__)
        6    0.000    0.000    0.022    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.022    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.020    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.020    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.019    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.018    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.001    0.001    0.016    0.016 surrogate.py:742(_coorbital_to_inertial_frame)
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
        1    0.005    0.005    0.005    0.005 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.005    0.005 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.004    0.004    0.005    0.005 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       66    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
        2    0.000    0.000    0.004    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.004    0.002 spline_interp_Cwrapper.py:50(interpolate)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
       11    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         28682 function calls in 0.025 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.026    0.026 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.025    0.025 surrogate.py:1721(__call__)
        1    0.000    0.000    0.025    0.025 surrogate.py:934(__call__)
        6    0.000    0.000    0.021    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.021    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.021    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.020    0.000 nodeFunction.py:205(__call__)
       66    0.000    0.000    0.020    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.018    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.018    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.018    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.011    0.000 validation.py:2793(validate_data)
       66    0.000    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       66    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      198    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
      396    0.000    0.000    0.001    0.000 _array_api.py:331(get_namespace)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         28686 function calls in 0.055 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.055    0.055 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.055    0.055 surrogate.py:1721(__call__)
        1    0.000    0.000    0.051    0.051 surrogate.py:934(__call__)
        1    0.001    0.001    0.028    0.028 surrogate.py:742(_coorbital_to_inertial_frame)
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
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         28682 function calls in 0.026 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.026    0.026 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.026    0.026 surrogate.py:1721(__call__)
        1    0.000    0.000    0.026    0.026 surrogate.py:934(__call__)
        6    0.000    0.000    0.021    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.021    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.021    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.020    0.000 nodeFunction.py:205(__call__)
       66    0.000    0.000    0.020    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.018    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.018    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.018    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.004    0.004 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      132    0.000    0.000    0.001    0.000 _py_warnings.py:294(simplefilter)
      198    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
      396    0.000    0.000    0.001    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         28669 function calls in 0.027 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.027    0.027 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.027    0.027 surrogate.py:1721(__call__)
        1    0.000    0.000    0.027    0.027 surrogate.py:934(__call__)
        6    0.000    0.000    0.021    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.021    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.021    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.020    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.020    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.018    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.018    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.018    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.000    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.005    0.005 surrogate.py:742(_coorbital_to_inertial_frame)
       66    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
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
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      132    0.000    0.000    0.001    0.000 _py_warnings.py:294(simplefilter)
     3894    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         28669 function calls in 0.190 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.190    0.190 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.011    0.011    0.190    0.190 surrogate.py:1721(__call__)
        1    0.000    0.000    0.172    0.172 surrogate.py:934(__call__)
        1    0.006    0.006    0.150    0.150 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.061    0.061    0.061    0.061 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.041    0.041 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.041    0.041    0.041    0.041 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.037    0.019 surrogate.py:91(_splinterp_Cwrapper)
        2    0.037    0.018    0.037    0.019 spline_interp_Cwrapper.py:50(interpolate)
        6    0.000    0.000    0.022    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.022    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.020    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.020    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.019    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
        5    0.007    0.001    0.007    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         28669 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.025    0.025 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.025    0.025 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 surrogate.py:934(__call__)
        6    0.000    0.000    0.021    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.021    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.021    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.020    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.020    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.018    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.018    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.018    0.000 evaluate_fit.py:128(GPR_predict)
      132    0.001    0.000    0.011    0.000 validation.py:2793(validate_data)
       66    0.000    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.009    0.000 validation.py:725(check_array)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       66    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
        1    0.000    0.000    0.003    0.003 surrogate.py:742(_coorbital_to_inertial_frame)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
        2    0.000    0.000    0.001    0.000 surrogate.py:91(_splinterp_Cwrapper)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
        2    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
     3894    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
      198    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         28669 function calls in 0.057 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.057    0.057 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.057    0.057 surrogate.py:1721(__call__)
        1    0.000    0.000    0.055    0.055 surrogate.py:934(__call__)
        1    0.001    0.001    0.032    0.032 surrogate.py:742(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.022    0.004 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.022    0.004 surrogate.py:417(__call__)
       12    0.000    0.000    0.022    0.002 surrogate.py:292(__call__)
       66    0.000    0.000    0.020    0.000 nodeFunction.py:205(__call__)
       66    0.001    0.000    0.020    0.000 nodeFunction.py:140(__call__)
       66    0.000    0.000    0.019    0.000 nodeFunction.py:96(__call__)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:247(gprfitEvaluator)
       66    0.000    0.000    0.019    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.012    0.012    0.012    0.012 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      132    0.001    0.000    0.012    0.000 validation.py:2793(validate_data)
       66    0.001    0.000    0.011    0.000 _gpr.py:373(predict)
      132    0.001    0.000    0.009    0.000 validation.py:725(check_array)
        1    0.000    0.000    0.009    0.009 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.009    0.009 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.008    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
       66    0.000    0.000    0.007    0.000 _base.py:297(predict)
       66    0.000    0.000    0.007    0.000 _base.py:287(_decision_function)
       66    0.000    0.000    0.005    0.000 kernels.py:833(__call__)
      132    0.001    0.000    0.004    0.000 validation.py:103(_assert_all_finite)
       66    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      396    0.001    0.000    0.003    0.000 validation.py:371(_num_samples)
       66    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     2576    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       66    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      198    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      132    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       66    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      198    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      198    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      132    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      264    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         94030 function calls in 0.130 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.131    0.131 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.130    0.130 surrogate.py:1721(__call__)
        1    0.002    0.002    0.126    0.126 surrogate.py:934(__call__)
       11    0.000    0.000    0.097    0.009 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.097    0.009 surrogate.py:417(__call__)
       19    0.000    0.000    0.097    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.088    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.087    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.080    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.080    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.080    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.002    0.000    0.049    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.047    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.040    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.030    0.000 _base.py:297(predict)
      218    0.002    0.000    0.030    0.000 _base.py:287(_decision_function)
        1    0.001    0.001    0.026    0.026 surrogate.py:742(_coorbital_to_inertial_frame)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      436    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
       23    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      218    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      436    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
        2    0.000    0.000    0.005    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
      654    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      218    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      654    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      436    0.001    0.000    0.004    0.000 fromnumeric.py:2304(sum)
    12862    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      654    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         94026 function calls in 0.103 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.103    0.103 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.103    0.103 surrogate.py:1721(__call__)
        1    0.002    0.002    0.103    0.103 surrogate.py:934(__call__)
       11    0.000    0.000    0.093    0.008 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.093    0.008 surrogate.py:417(__call__)
       19    0.000    0.000    0.093    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.084    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.077    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.002    0.000    0.047    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.029    0.000 _base.py:297(predict)
      218    0.001    0.000    0.029    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
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
      436    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
      654    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     1308    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      436    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
     1744    0.001    0.000    0.003    0.000 _config.py:35(get_config)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      436    0.000    0.000    0.003    0.000 _base.py:711(__sklearn_tags__)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         94030 function calls in 0.150 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.150    0.150 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.150    0.150 surrogate.py:1721(__call__)
        1    0.002    0.002    0.143    0.143 surrogate.py:934(__call__)
       11    0.000    0.000    0.092    0.008 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.092    0.008 surrogate.py:417(__call__)
       19    0.000    0.000    0.092    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.084    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.083    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.077    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.077    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.076    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.002    0.002    0.048    0.048 surrogate.py:742(_coorbital_to_inertial_frame)
      436    0.002    0.000    0.047    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.045    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.038    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.029    0.000 _base.py:297(predict)
      218    0.001    0.000    0.029    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.020    0.020 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.019    0.019    0.020    0.020 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
        1    0.016    0.016    0.016    0.016 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
        2    0.000    0.000    0.008    0.004 surrogate.py:91(_splinterp_Cwrapper)
        2    0.008    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      218    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      436    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       18    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
      654    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      218    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      654    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    12862    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         94026 function calls in 0.103 seconds

   Ordered by: cumulative time
   List reduced from 176 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.104    0.104 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.104    0.104 surrogate.py:1721(__call__)
        1    0.002    0.002    0.103    0.103 surrogate.py:934(__call__)
       11    0.000    0.000    0.092    0.008 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.092    0.008 surrogate.py:417(__call__)
       19    0.000    0.000    0.092    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.084    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.083    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.077    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.077    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.076    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.002    0.000    0.047    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.045    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.038    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.029    0.000 _base.py:297(predict)
      218    0.001    0.000    0.029    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
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
      654    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      436    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
     1308    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      436    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
     1744    0.001    0.000    0.003    0.000 _config.py:35(get_config)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         94006 function calls in 0.104 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.105    0.105 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.105    0.105 surrogate.py:1721(__call__)
        1    0.002    0.002    0.104    0.104 surrogate.py:934(__call__)
       11    0.000    0.000    0.093    0.008 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.093    0.008 surrogate.py:417(__call__)
       19    0.000    0.000    0.093    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.084    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.084    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.077    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.002    0.000    0.047    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.029    0.000 _base.py:297(predict)
      218    0.001    0.000    0.029    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      218    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      436    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      654    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      218    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      654    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
     1308    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
    12862    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      436    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      654    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      436    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
     1744    0.001    0.000    0.003    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         94006 function calls in 0.350 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.351    0.351 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.020    0.020    0.351    0.351 surrogate.py:1721(__call__)
        1    0.003    0.003    0.321    0.321 surrogate.py:934(__call__)
        1    0.008    0.008    0.223    0.223 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.095    0.009 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.095    0.009 surrogate.py:417(__call__)
       19    0.000    0.000    0.094    0.005 surrogate.py:292(__call__)
        1    0.000    0.000    0.093    0.093 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.092    0.092    0.093    0.093 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.086    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.085    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.079    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.079    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.078    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.070    0.070    0.070    0.070 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      436    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.047    0.000 _gpr.py:373(predict)
        2    0.000    0.000    0.045    0.023 surrogate.py:91(_splinterp_Cwrapper)
        2    0.045    0.023    0.045    0.023 spline_interp_Cwrapper.py:50(interpolate)
      436    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.030    0.000 _base.py:297(predict)
      218    0.002    0.000    0.030    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.021    0.000 kernels.py:833(__call__)
      218    0.001    0.000    0.016    0.000 kernels.py:931(__call__)
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      218    0.005    0.000    0.010    0.000 kernels.py:1525(__call__)
        7    0.010    0.001    0.010    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      218    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      654    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      436    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      654    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      218    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
      654    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      872    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    12862    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         94006 function calls in 0.101 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.102    0.102 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.102    0.102 surrogate.py:1721(__call__)
        1    0.002    0.002    0.102    0.102 surrogate.py:934(__call__)
       11    0.000    0.000    0.093    0.008 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.093    0.008 surrogate.py:417(__call__)
       19    0.000    0.000    0.093    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.084    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.077    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.077    0.000 evaluate_fit.py:128(GPR_predict)
      436    0.002    0.000    0.047    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.030    0.000 _base.py:297(predict)
      218    0.001    0.000    0.029    0.000 _base.py:287(_decision_function)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
      218    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      218    0.004    0.000    0.009    0.000 kernels.py:1525(__call__)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
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
      654    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      436    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      436    0.001    0.000    0.003    0.000 fromnumeric.py:2304(sum)
     1744    0.001    0.000    0.003    0.000 _config.py:35(get_config)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
      436    0.000    0.000    0.003    0.000 _base.py:711(__sklearn_tags__)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         94006 function calls in 0.155 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.156    0.156 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.156    0.156 surrogate.py:1721(__call__)
        1    0.002    0.002    0.152    0.152 surrogate.py:934(__call__)
       11    0.000    0.000    0.094    0.009 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.094    0.009 surrogate.py:417(__call__)
       19    0.000    0.000    0.094    0.005 surrogate.py:292(__call__)
      218    0.000    0.000    0.085    0.000 nodeFunction.py:205(__call__)
      218    0.002    0.000    0.085    0.000 nodeFunction.py:110(__call__)
      218    0.000    0.000    0.078    0.000 nodeFunction.py:96(__call__)
      218    0.000    0.000    0.078    0.000 evaluate_fit.py:247(gprfitEvaluator)
      218    0.001    0.000    0.078    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.002    0.002    0.055    0.055 surrogate.py:742(_coorbital_to_inertial_frame)
      436    0.002    0.000    0.048    0.000 validation.py:2793(validate_data)
      218    0.002    0.000    0.046    0.000 _gpr.py:373(predict)
      436    0.005    0.000    0.039    0.000 validation.py:725(check_array)
      218    0.000    0.000    0.029    0.000 _base.py:297(predict)
      218    0.001    0.000    0.029    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.022    0.022 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.022    0.022 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
        1    0.019    0.019    0.019    0.019 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.001    0.000    0.016    0.000 kernels.py:931(__call__)
      436    0.004    0.000    0.015    0.000 validation.py:103(_assert_all_finite)
     1308    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      218    0.005    0.000    0.010    0.000 kernels.py:1525(__call__)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
     8504    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
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
      654    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
```

#### PR-72

##### NRSur7dq4 / mks_dt_0.000244140625_flow_0

```text
         27100 function calls in 0.030 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.030    0.030 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.030    0.030 surrogate.py:1721(__call__)
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:847(inertial_waveform_modes)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.007    0.007 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:626(_integrate_forward)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2806    0.001    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
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
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         27101 function calls in 0.030 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.030    0.030 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.030    0.030 surrogate.py:1721(__call__)
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:110(rotateWaveform)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.006    0.006    0.008    0.008 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:626(_integrate_forward)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:660(_integrate_backward)
     1633    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
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
        1    0.000    0.000    0.034    0.034 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.034    0.034 surrogate.py:1721(__call__)
        1    0.000    0.000    0.033    0.033 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:110(rotateWaveform)
        1    0.007    0.007    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
     1633    0.002    0.000    0.002    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.000    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         27101 function calls in 0.033 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.033    0.033 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.033    0.033 surrogate.py:1721(__call__)
        1    0.000    0.000    0.033    0.033 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.010    0.010 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:110(rotateWaveform)
        1    0.007    0.007    0.009    0.009 precessing_surrogate.py:42(_wignerD_matrices)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:626(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2806    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.003    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        4    0.001    0.000    0.002    0.001 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
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
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
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
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      238    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        4    0.000    0.000    0.006    0.002 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.006    0.006 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.006    0.006    0.006    0.006 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
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
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         27707 function calls in 0.029 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.029    0.029 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.029    0.029 surrogate.py:1721(__call__)
        1    0.000    0.000    0.029    0.029 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.004    0.004    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2876    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
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
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         20326 function calls in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 91 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.022    0.022 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.022    0.022 surrogate.py:1721(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      238    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.004    0.004    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
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
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         27707 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.004    0.004    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     2876    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.000    0.000    0.002    0.000 surrogate.py:106(_splinterp_Cwrapper_many)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     1703    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         43096 function calls in 0.045 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.045    0.045 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.045    0.045 surrogate.py:1721(__call__)
        1    0.000    0.000    0.045    0.045 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.045    0.045 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
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
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      675    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         43097 function calls in 0.045 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.044    0.044 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.044    0.044 surrogate.py:1721(__call__)
        1    0.000    0.000    0.044    0.044 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.044    0.044 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5135    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
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
         43096 function calls in 0.048 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.048    0.048 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.048    0.048 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.020    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.015    0.015 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
     5135    0.003    0.000    0.013    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 surrogate.py:86(_splinterp_Cwrapper_many_complex)
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
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
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
         43097 function calls in 0.053 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.053    0.053 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.053    0.053 surrogate.py:1721(__call__)
        1    0.000    0.000    0.052    0.052 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.052    0.052 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:392(__call__)
      546    0.002    0.000    0.023    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.019    0.019 precessing_surrogate.py:626(_integrate_forward)
     1638    0.003    0.000    0.018    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5135    0.003    0.000    0.015    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.015    0.015 precessing_surrogate.py:110(rotateWaveform)
        1    0.010    0.010    0.012    0.012 precessing_surrogate.py:42(_wignerD_matrices)
     5135    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:660(_integrate_backward)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
     5135    0.002    0.000    0.004    0.000 precessing_surrogate.py:1191(<lambda>)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
     2605    0.002    0.000    0.002    0.000 {built-in method numpy.array}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     5135    0.002    0.000    0.002    0.000 surrogate.py:2588(get_fit_settings)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
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
        1    0.001    0.001    0.020    0.020 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.018    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.003    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     4637    0.003    0.000    0.012    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
        4    0.000    0.000    0.011    0.003 surrogate.py:106(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     4637    0.001    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
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
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:155(<genexpr>)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         44452 function calls in 0.047 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.046    0.046 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5274    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:626(_integrate_forward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:660(_integrate_backward)
     5274    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
        4    0.000    0.000    0.004    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:345(_get_t_from_omega)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:339(get_omega)
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

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         35075 function calls in 0.039 seconds

   Ordered by: cumulative time
   List reduced from 94 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.039    0.039 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.039    0.039 surrogate.py:1721(__call__)
        1    0.000    0.000    0.039    0.039 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.039    0.039 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.020    0.020 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.019    0.019 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.018    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.002    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     4637    0.003    0.000    0.012    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.007    0.000    0.007    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
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
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
       96    0.000    0.000    0.000    0.000 __init__.py:613(cast)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         44452 function calls in 0.045 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.045    0.045 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.045    0.045 surrogate.py:1721(__call__)
        1    0.000    0.000    0.045    0.045 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.045    0.045 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.019    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.015    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     5274    0.003    0.000    0.014    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.002    0.002    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:626(_integrate_forward)
        1    0.001    0.001    0.010    0.010 precessing_surrogate.py:660(_integrate_backward)
     5274    0.008    0.000    0.008    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:91(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:106(_splinterp_Cwrapper_many)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:345(_get_t_from_omega)
      268    0.000    0.000    0.002    0.000 precessing_surrogate.py:339(get_omega)
        1    0.000    0.000    0.001    0.001 surrogate.py:86(_splinterp_Cwrapper_many_complex)
     2744    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         9044 function calls in 0.061 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.061    0.061 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.061    0.061 surrogate.py:1721(__call__)
        1    0.003    0.003    0.056    0.056 surrogate.py:934(__call__)
        1    0.001    0.001    0.033    0.033 surrogate.py:742(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.020    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.020    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.015    0.015 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.014    0.014    0.015    0.015 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.005    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.005    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       21    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        3    0.001    0.000    0.001    0.000 {built-in method numpy.ascontiguousarray}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        9    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
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
        1    0.001    0.001    0.007    0.007 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.002    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         9044 function calls in 0.101 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.101    0.101 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.101    0.101 surrogate.py:1721(__call__)
        1    0.003    0.003    0.079    0.079 surrogate.py:934(__call__)
        1    0.003    0.003    0.056    0.056 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.022    0.022    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.020    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.020    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.017    0.017 {method 'update' of 'dict' objects}
       21    0.017    0.001    0.017    0.001 surrogate.py:2126(<genexpr>)
        1    0.017    0.017    0.017    0.017 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.010    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
       26    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        9    0.003    0.000    0.003    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        4    0.002    0.001    0.002    0.001 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      177    0.000    0.000    0.000    0.000 enum.py:187(__get__)
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
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.011    0.000 nodeFunction.py:125(__call__)
        1    0.001    0.001    0.008    0.008 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       26    0.006    0.000    0.006    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.006    0.000    0.006    0.000 evaluate_fit.py:128(GPR_predict_fast)
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
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       21    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
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
       12    0.000    0.000    0.019    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        1    0.001    0.001    0.009    0.009 surrogate.py:742(_coorbital_to_inertial_frame)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
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
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         9017 function calls in 0.303 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.303    0.303 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.026    0.026    0.303    0.303 surrogate.py:1721(__call__)
        1    0.003    0.003    0.265    0.265 surrogate.py:934(__call__)
        1    0.010    0.010    0.241    0.241 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.102    0.102 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.102    0.102    0.102    0.102 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.074    0.074    0.074    0.074 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.047    0.023 surrogate.py:91(_splinterp_Cwrapper)
        2    0.046    0.023    0.047    0.023 spline_interp_Cwrapper.py:50(interpolate)
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.019    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        9    0.012    0.001    0.012    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
      177    0.001    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
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
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      177    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        9    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
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
        1    0.002    0.002    0.059    0.059 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.026    0.026 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.025    0.025    0.026    0.026 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.020    0.020    0.020    0.020 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       12    0.000    0.000    0.020    0.002 surrogate.py:425(_eval_sur)
       12    0.000    0.000    0.020    0.002 surrogate.py:417(__call__)
       22    0.000    0.000    0.019    0.001 surrogate.py:292(__call__)
      177    0.000    0.000    0.012    0.000 nodeFunction.py:220(__call__)
      177    0.001    0.000    0.012    0.000 nodeFunction.py:125(__call__)
        2    0.000    0.000    0.010    0.005 surrogate.py:91(_splinterp_Cwrapper)
        2    0.009    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
      177    0.000    0.000    0.007    0.000 nodeFunction.py:111(__call__)
      177    0.000    0.000    0.007    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      177    0.006    0.000    0.007    0.000 evaluate_fit.py:128(GPR_predict_fast)
       26    0.007    0.000    0.007    0.000 {method 'dot' of 'numpy.ndarray' objects}
      177    0.001    0.000    0.002    0.000 _py_warnings.py:254(filterwarnings)
        9    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      177    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      177    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      177    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      177    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      177    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      177    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      531    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      531    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      177    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      177    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
      885    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      887    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
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
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
       11    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         3536 function calls in 0.008 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.008    0.008 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.008    0.008 surrogate.py:1721(__call__)
        1    0.000    0.000    0.008    0.008 surrogate.py:934(__call__)
        6    0.000    0.000    0.004    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.004    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.004    0.000 surrogate.py:292(__call__)
        1    0.000    0.000    0.003    0.003 surrogate.py:742(_coorbital_to_inertial_frame)
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
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
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
        2    0.000    0.000    0.007    0.003 surrogate.py:91(_splinterp_Cwrapper)
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
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

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
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       11    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         3523 function calls in 0.010 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

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
        1    0.000    0.000    0.002    0.002 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.001    0.001 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         3523 function calls in 0.174 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.174    0.174 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.012    0.012    0.174    0.174 surrogate.py:1721(__call__)
        1    0.000    0.000    0.155    0.155 surrogate.py:934(__call__)
        1    0.006    0.006    0.150    0.150 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.061    0.061    0.061    0.061 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        1    0.000    0.000    0.041    0.041 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.041    0.041    0.041    0.041 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.000    0.000    0.037    0.019 surrogate.py:91(_splinterp_Cwrapper)
        2    0.037    0.018    0.037    0.018 spline_interp_Cwrapper.py:50(interpolate)
        5    0.007    0.001    0.007    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.005    0.001 surrogate.py:425(_eval_sur)
        6    0.000    0.000    0.005    0.001 surrogate.py:417(__call__)
       12    0.000    0.000    0.005    0.000 surrogate.py:292(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:220(__call__)
       66    0.000    0.000    0.003    0.000 nodeFunction.py:155(__call__)
       66    0.000    0.000    0.002    0.000 nodeFunction.py:111(__call__)
       66    0.000    0.000    0.002    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
        2    0.002    0.001    0.002    0.001 surrogate.py:732(_search_omega)
        4    0.002    0.000    0.002    0.000 _function_base_impl.py:1402(diff)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
        2    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       20    0.000    0.000    0.000    0.000 __init__.py:613(cast)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         3523 function calls in 0.008 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.007    0.007 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.007    0.007 surrogate.py:1721(__call__)
        1    0.000    0.000    0.007    0.007 surrogate.py:934(__call__)
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
        2    0.000    0.000    0.001    0.000 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:50(interpolate)
       16    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       66    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:320(_add_filter)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
       66    0.000    0.000    0.000    0.000 einsumfunc.py:1244(einsum)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:668(__exit__)
       66    0.000    0.000    0.000    0.000 _py_warnings.py:639(__enter__)
       66    0.000    0.000    0.000    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
       66    0.000    0.000    0.000    0.000 __init__.py:287(compile)
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        5    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
      330    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         3523 function calls in 0.040 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.039    0.039 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.039    0.039 surrogate.py:1721(__call__)
        1    0.000    0.000    0.037    0.037 surrogate.py:934(__call__)
        1    0.001    0.001    0.032    0.032 surrogate.py:742(_coorbital_to_inertial_frame)
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
       66    0.001    0.000    0.002    0.000 evaluate_fit.py:128(GPR_predict_fast)
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
       66    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      198    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      198    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        1    0.000    0.000    0.000    0.000 surrogate.py:1634(_check_params)
       66    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
       66    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       16    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         10972 function calls in 0.056 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.056    0.056 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.001    0.001    0.056    0.056 surrogate.py:1721(__call__)
        1    0.002    0.002    0.052    0.052 surrogate.py:934(__call__)
        1    0.001    0.001    0.025    0.025 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
        1    0.000    0.000    0.010    0.010 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.009    0.009    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.008    0.000    0.009    0.000 evaluate_fit.py:128(GPR_predict_fast)
        1    0.008    0.008    0.008    0.008 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.005    0.002 surrogate.py:91(_splinterp_Cwrapper)
        2    0.004    0.002    0.005    0.002 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       18    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
        7    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         10968 function calls in 0.033 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.033    0.033 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.033    0.033 surrogate.py:1721(__call__)
        1    0.002    0.002    0.032    0.032 surrogate.py:934(__call__)
       11    0.000    0.000    0.023    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.014    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.009    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.007    0.000    0.008    0.000 evaluate_fit.py:128(GPR_predict_fast)
       23    0.008    0.000    0.008    0.000 {method 'dot' of 'numpy.ndarray' objects}
        1    0.001    0.001    0.006    0.006 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.003    0.003 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      218    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
        2    0.000    0.000    0.001    0.001 surrogate.py:91(_splinterp_Cwrapper)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      218    0.001    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
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
        1    0.002    0.002    0.075    0.075 surrogate.py:934(__call__)
        1    0.002    0.002    0.049    0.049 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.020    0.020 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.020    0.020    0.020    0.020 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.016    0.016    0.016    0.016 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.015    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.015    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.009    0.000 nodeFunction.py:111(__call__)
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
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
        3    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         10968 function calls in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 100 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1721(__call__)
        1    0.002    0.002    0.037    0.037 surrogate.py:934(__call__)
       11    0.000    0.000    0.027    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.027    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.026    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.018    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.018    0.000 nodeFunction.py:125(__call__)
      218    0.000    0.000    0.012    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.012    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
      218    0.011    0.000    0.012    0.000 evaluate_fit.py:128(GPR_predict_fast)
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
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       18    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         10948 function calls in 0.035 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.035    0.035 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.000    0.000    0.035    0.035 surrogate.py:1721(__call__)
        1    0.002    0.002    0.035    0.035 surrogate.py:934(__call__)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
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
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         10948 function calls in 0.284 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.284    0.284 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.021    0.021    0.284    0.284 surrogate.py:1721(__call__)
        1    0.003    0.003    0.253    0.253 surrogate.py:934(__call__)
        1    0.009    0.009    0.225    0.225 surrogate.py:742(_coorbital_to_inertial_frame)
        1    0.000    0.000    0.093    0.093 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.093    0.093    0.093    0.093 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.070    0.070    0.070    0.070 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
        2    0.000    0.000    0.045    0.022 surrogate.py:91(_splinterp_Cwrapper)
        2    0.044    0.022    0.045    0.022 spline_interp_Cwrapper.py:50(interpolate)
       11    0.000    0.000    0.025    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
      218    0.002    0.000    0.016    0.000 nodeFunction.py:125(__call__)
        7    0.010    0.001    0.010    0.001 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.010    0.000 nodeFunction.py:111(__call__)
      218    0.000    0.000    0.010    0.000 evaluate_fit.py:281(gprfastfitEvaluator)
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
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 surrogate.py:918(_TaylorT3_phase_22)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
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
       11    0.000    0.000    0.023    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.023    0.001 surrogate.py:292(__call__)
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
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
        4    0.000    0.000    0.000    0.000 _function_base_impl.py:1402(diff)
        2    0.000    0.000    0.000    0.000 surrogate.py:732(_search_omega)
        7    0.000    0.000    0.000    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.ascontiguousarray}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
      218    0.000    0.000    0.000    0.000 enum.py:187(__get__)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         10948 function calls in 0.087 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.087    0.087 benchmark_surrogate_evaluations.py:346(evaluate_case)
        1    0.002    0.002    0.087    0.087 surrogate.py:1721(__call__)
        1    0.002    0.002    0.083    0.083 surrogate.py:934(__call__)
        1    0.002    0.002    0.056    0.056 surrogate.py:742(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.024    0.002 surrogate.py:425(_eval_sur)
       11    0.000    0.000    0.024    0.002 surrogate.py:417(__call__)
       19    0.000    0.000    0.024    0.001 surrogate.py:292(__call__)
        1    0.000    0.000    0.023    0.023 surrogate.py:86(_splinterp_Cwrapper_many_complex)
        1    0.023    0.023    0.023    0.023 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.019    0.019    0.019    0.019 {built-in method gwsurrogate.precessing_utils._utils.coorbital_to_inertial_in_place}
      218    0.000    0.000    0.016    0.000 nodeFunction.py:220(__call__)
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
        1    0.001    0.001    0.001    0.001 {built-in method numpy.zeros}
      218    0.000    0.000    0.001    0.000 einsumfunc.py:1244(einsum)
      218    0.001    0.000    0.001    0.000 {built-in method numpy._core._multiarray_umath.c_einsum}
      218    0.000    0.000    0.001    0.000 _py_warnings.py:639(__enter__)
        2    0.001    0.000    0.001    0.000 surrogate.py:732(_search_omega)
      218    0.000    0.000    0.001    0.000 _py_warnings.py:668(__exit__)
      218    0.000    0.000    0.001    0.000 __init__.py:287(compile)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
      218    0.000    0.000    0.000    0.000 __init__.py:330(_compile)
      654    0.000    0.000    0.000    0.000 warnings.py:80(__enter__)
      654    0.000    0.000    0.000    0.000 warnings.py:84(__exit__)
      218    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      218    0.000    0.000    0.000    0.000 _py_warnings.py:111(_get_filters)
        1    0.000    0.000    0.000    0.000 {built-in method numpy.arange}
     1092    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 _function_base_impl.py:5577(append)
       23    0.000    0.000    0.000    0.000 {built-in method numpy.array}
     1090    0.000    0.000    0.000    0.000 einsumfunc.py:1236(_einsum_dispatcher)
```
