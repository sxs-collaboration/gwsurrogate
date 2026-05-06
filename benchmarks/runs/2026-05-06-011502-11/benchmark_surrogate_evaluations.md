# GWSurrogate Evaluation Timing

Generated: 2026-05-06T01:14:58.924438+00:00

Times below are seconds per model evaluation. Raw repeats and context are in the JSON output.

PNG timing table: `test/benchmark_surrogate_evaluations.png`

## Summary

### master

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0585236` s, median `0.0587595` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0468246` s, median `0.0471703` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0855533` s, median `0.0856492` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0654381` s, median `0.065688` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.216043` s, median `0.216784` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0691161` s, median `0.0695275` s
- `dt=0.5 M`, `f_low=0`: best `0.0615371` s, median `0.0618925` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0369389` s, median `0.0371351` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0914549` s, median `0.0918544` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0711302` s, median `0.0714067` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.138662` s, median `0.139059` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0968771` s, median `0.0970859` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.377563` s, median `0.37913` s
- `dt=0.1 M`, `f_low=0.01`: best `0.107117` s, median `0.107271` s
- `dt=0.5 M`, `f_low=0`: best `0.105615` s, median `0.105745` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0578822` s, median `0.0581555` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.421947` s, median `0.423141` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.143458` s, median `0.144032` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.726775` s, median `0.727167` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.161664` s, median `0.161984` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.172704` s, median `0.173011` s
- `dt=0.1 M`, `f_low=0.002`: best `3.63527` s, median `3.63882` s
- `dt=0.5 M`, `f_low=0.01`: best `0.127341` s, median `0.127708` s
- `dt=0.5 M`, `f_low=0.002`: best `0.826527` s, median `0.826904` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.160106` s, median `0.16088` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.032951` s, median `0.0329745` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.294486` s, median `0.295939` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0402675` s, median `0.0403818` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0447402` s, median `0.0447705` s
- `dt=0.1 M`, `f_low=0.002`: best `1.60534` s, median `1.60726` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0283852` s, median `0.0284667` s
- `dt=0.5 M`, `f_low=0.002`: best `0.335228` s, median `0.336362` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.394235` s, median `0.395826` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.13725` s, median `0.137474` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.657481` s, median `0.65849` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.156612` s, median `0.156836` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.167346` s, median `0.167549` s
- `dt=0.1 M`, `f_low=0.002`: best `3.25509` s, median `3.26333` s
- `dt=0.5 M`, `f_low=0.01`: best `0.123034` s, median `0.123165` s
- `dt=0.5 M`, `f_low=0.002`: best `0.748293` s, median `0.748907` s

### PR-70

#### NRSur7dq4

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0189016` s, median `0.0190038` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0188539` s, median `0.0190724` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0215006` s, median `0.0217383` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0210549` s, median `0.0210957` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0188165` s, median `0.0189524` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0179127` s, median `0.0181323` s
- `dt=0.5 M`, `f_low=0`: best `0.0145698` s, median `0.0146274` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0171141` s, median `0.0171653` s

#### SEOBNRv4PHMSur

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=0 Hz`: best `0.0290045` s, median `0.0294129` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0285478` s, median `0.0286512` s
- `dt=1/8192 s`, `f_low=0 Hz`: best `0.0303149` s, median `0.0307732` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0292146` s, median `0.0294493` s

Geometric Units:

- `dt=0.1 M`, `f_low=0`: best `0.0330837` s, median `0.0333631` s
- `dt=0.1 M`, `f_low=0.01`: best `0.0297054` s, median `0.0297914` s
- `dt=0.5 M`, `f_low=0`: best `0.0254921` s, median `0.0257904` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0285109` s, median `0.0287098` s

#### NRHybSur3dq8

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.157645` s, median `0.158032` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0650787` s, median `0.0653402` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.209349` s, median `0.209789` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0700698` s, median `0.0701824` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.071764` s, median `0.0720434` s
- `dt=0.1 M`, `f_low=0.002`: best `0.735739` s, median `0.736833` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0632495` s, median `0.0633873` s
- `dt=0.5 M`, `f_low=0.002`: best `0.220376` s, median `0.221762` s

#### NRHybSur2dq15

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.0404044` s, median `0.0407241` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0149899` s, median `0.0150011` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.0686172` s, median `0.0692027` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0164496` s, median `0.0165032` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0166619` s, median `0.016743` s
- `dt=0.1 M`, `f_low=0.002`: best `0.334245` s, median `0.334374` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0142007` s, median `0.0143104` s
- `dt=0.5 M`, `f_low=0.002`: best `0.0747171` s, median `0.0749418` s

#### NRHybSur3dq8_CCE

MKS Units (M_tot = 70 M_sun):

- `dt=1/4096 s`, `f_low=7 Hz`: best `0.133017` s, median `0.133391` s
- `dt=1/4096 s`, `f_low=20 Hz`: best `0.0683091` s, median `0.0689379` s
- `dt=1/8192 s`, `f_low=7 Hz`: best `0.186831` s, median `0.187035` s
- `dt=1/8192 s`, `f_low=20 Hz`: best `0.0718216` s, median `0.0721031` s

Geometric Units:

- `dt=0.1 M`, `f_low=0.01`: best `0.0744138` s, median `0.0745134` s
- `dt=0.1 M`, `f_low=0.002`: best `0.649873` s, median `0.651103` s
- `dt=0.5 M`, `f_low=0.01`: best `0.0664424` s, median `0.0666371` s
- `dt=0.5 M`, `f_low=0.002`: best `0.204119` s, median `0.20443` s

## Context

### master

- Git branch: `master`
- Git commit: `a84a5da1aa62624dd73c52103ab7fab6410bb32a`
- Git describe: `v1.1.8-11-ga84a5da`
- Python: `3.14.4 (main, Apr  8 2026, 02:27:22) [GCC 13.3.0]`
- Platform: `Linux 6.17.0-1010-azure x86_64`
- CPU count: `4`
- Conda env: `unknown`

### PR-70

- Git branch: `unknown`
- Git commit: `76f32b025c1f7842b5cd0c24b773e7b6d3462a7a`
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
Address sizes:                           46 bits physical, 57 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  4
On-line CPU(s) list:                     0-3
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
CPU family:                              6
Model:                                   106
Thread(s) per core:                      2
Core(s) per socket:                      2
Socket(s):                               1
Stepping:                                6
CPU(s) scaling MHz:                      113%
CPU max MHz:                             2800.0000
CPU min MHz:                             800.0000
BogoMIPS:                                5586.87
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch tpr_shadow ept vpid ept_ad fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves vnmi avx512vbmi umip avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm arch_capabilities
Virtualization:                          VT-x
Hypervisor vendor:                       Microsoft
Virtualization type:                     full
L1d cache:                               96 KiB (2 instances)
L1i cache:                               64 KiB (2 instances)
L2 cache:                                2.5 MiB (2 instances)
L3 cache:                                48 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-3
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Mitigation; Aligned branch/return thunks
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Vulnerable
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Vulnerable
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Retpolines; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Retpoline
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Not affected
```

#### PR-70

lscpu:

```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 57 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  4
On-line CPU(s) list:                     0-3
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
CPU family:                              6
Model:                                   106
Thread(s) per core:                      2
Core(s) per socket:                      2
Socket(s):                               1
Stepping:                                6
CPU(s) scaling MHz:                      106%
CPU max MHz:                             2800.0000
CPU min MHz:                             800.0000
BogoMIPS:                                5586.87
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch tpr_shadow ept vpid ept_ad fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves vnmi avx512vbmi umip avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid fsrm arch_capabilities
Virtualization:                          VT-x
Hypervisor vendor:                       Microsoft
Virtualization type:                     full
L1d cache:                               96 KiB (2 instances)
L1i cache:                               64 KiB (2 instances)
L2 cache:                                2.5 MiB (2 instances)
L3 cache:                                48 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-3
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Mitigation; Aligned branch/return thunks
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Vulnerable
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Vulnerable
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Retpolines; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Retpoline
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Not affected
```

### cProfile

#### master

##### NRSur7dq4 / mks_dt_0.000244140625_flow_0

```text
         28760 function calls in 0.069 seconds

   Ordered by: cumulative time
   List reduced from 95 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.069    0.069 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.069    0.069 surrogate.py:1721(__call__)
        1    0.001    0.001    0.068    0.068 precessing_surrogate.py:933(__call__)
      196    0.000    0.000    0.043    0.000 surrogate.py:85(_splinterp_Cwrapper)
      196    0.007    0.000    0.043    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.039    0.008 precessing_surrogate.py:849(splinterp_many)
      561    0.017    0.000    0.017    0.000 {built-in method builtins.max}
      560    0.016    0.000    0.016    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:387(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:105(rotateWaveform)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.005    0.005    0.007    0.007 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:621(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2806    0.002    0.000    0.004    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
     2806    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
     1638    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      784    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:655(_integrate_backward)
      784    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      588    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      784    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:340(_get_t_from_omega)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:334(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         28761 function calls in 0.059 seconds

   Ordered by: cumulative time
   List reduced from 95 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.059    0.059 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.059    0.059 surrogate.py:1721(__call__)
        1    0.001    0.001    0.059    0.059 precessing_surrogate.py:933(__call__)
      196    0.000    0.000    0.036    0.000 surrogate.py:85(_splinterp_Cwrapper)
      196    0.007    0.000    0.036    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.032    0.006 precessing_surrogate.py:849(splinterp_many)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:387(__call__)
      560    0.013    0.000    0.013    0.000 {built-in method builtins.min}
      561    0.013    0.000    0.013    0.000 {built-in method builtins.max}
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:105(rotateWaveform)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:621(_integrate_forward)
      837    0.001    0.000    0.005    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2806    0.002    0.000    0.004    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
     2806    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      784    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:655(_integrate_backward)
      784    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1638    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      588    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      784    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:340(_get_t_from_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:334(get_omega)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
      201    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         28760 function calls in 0.096 seconds

   Ordered by: cumulative time
   List reduced from 95 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.096    0.096 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.096    0.096 surrogate.py:1721(__call__)
        1    0.001    0.001    0.095    0.095 precessing_surrogate.py:933(__call__)
      196    0.000    0.000    0.069    0.000 surrogate.py:85(_splinterp_Cwrapper)
      196    0.010    0.000    0.069    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.065    0.013 precessing_surrogate.py:849(splinterp_many)
      561    0.028    0.000    0.028    0.000 {built-in method builtins.max}
      560    0.028    0.000    0.028    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:387(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:105(rotateWaveform)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.005    0.005    0.007    0.007 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:621(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2806    0.002    0.000    0.004    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
     2806    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     1638    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
      784    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:655(_integrate_backward)
      784    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      588    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      784    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:340(_get_t_from_omega)
      201    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       22    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:334(get_omega)
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         28761 function calls in 0.076 seconds

   Ordered by: cumulative time
   List reduced from 95 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.076    0.076 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.076    0.076 surrogate.py:1721(__call__)
        1    0.001    0.001    0.075    0.075 precessing_surrogate.py:933(__call__)
      196    0.000    0.000    0.050    0.000 surrogate.py:85(_splinterp_Cwrapper)
      196    0.008    0.000    0.050    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.046    0.009 precessing_surrogate.py:849(splinterp_many)
      561    0.020    0.000    0.020    0.000 {built-in method builtins.max}
      560    0.020    0.000    0.020    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:387(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:105(rotateWaveform)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.005    0.005    0.008    0.008 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:621(_integrate_forward)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2806    0.002    0.000    0.004    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
     2806    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
     1638    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
      784    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:655(_integrate_backward)
      784    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      588    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      784    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:340(_get_t_from_omega)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:334(get_omega)
      201    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         21557 function calls in 0.227 seconds

   Ordered by: cumulative time
   List reduced from 83 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.226    0.226 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.226    0.226 surrogate.py:1721(__call__)
        1    0.005    0.005    0.226    0.226 precessing_surrogate.py:933(__call__)
        5    0.000    0.000    0.199    0.040 precessing_surrogate.py:849(splinterp_many)
       53    0.000    0.000    0.195    0.004 surrogate.py:85(_splinterp_Cwrapper)
       53    0.020    0.000    0.195    0.004 spline_interp_Cwrapper.py:39(interpolate)
      262    0.087    0.000    0.087    0.000 {built-in method builtins.max}
      261    0.085    0.000    0.085    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:105(rotateWaveform)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:387(__call__)
        1    0.006    0.006    0.008    0.008 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.008    0.008 precessing_surrogate.py:621(_integrate_forward)
      238    0.001    0.000    0.007    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:179(_eval_vector_fit)
     1257    0.005    0.000    0.005    0.000 {built-in method numpy.array}
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2390    0.001    0.000    0.004    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
     2390    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
       58    0.002    0.000    0.002    0.000 {built-in method numpy.zeros}
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
      159    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      212    0.000    0.000    0.000    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:556(_initial_RK4)
      212    0.000    0.000    0.000    0.000 __init__.py:613(cast)
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:703(_assemble_mode_pair)
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
      212    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:827(rotate_spin)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:837(coorb_spins_from_copr_spins)
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         29367 function calls in 0.081 seconds

   Ordered by: cumulative time
   List reduced from 93 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.081    0.081 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.081    0.081 surrogate.py:1721(__call__)
        1    0.001    0.001    0.081    0.081 precessing_surrogate.py:933(__call__)
      196    0.000    0.000    0.057    0.000 surrogate.py:85(_splinterp_Cwrapper)
      196    0.008    0.000    0.057    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.053    0.011 precessing_surrogate.py:849(splinterp_many)
      561    0.023    0.000    0.023    0.000 {built-in method builtins.max}
      560    0.023    0.000    0.023    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.015    0.015 precessing_surrogate.py:387(__call__)
      279    0.001    0.000    0.008    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:105(rotateWaveform)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
      837    0.001    0.000    0.006    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2876    0.002    0.000    0.005    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:621(_integrate_forward)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:655(_integrate_backward)
     2876    0.003    0.000    0.003    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      784    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      784    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1708    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:340(_get_t_from_omega)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      588    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      784    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
      201    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         21557 function calls in 0.070 seconds

   Ordered by: cumulative time
   List reduced from 83 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.070    0.070 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.070    0.070 surrogate.py:1721(__call__)
        1    0.001    0.001    0.070    0.070 precessing_surrogate.py:933(__call__)
        5    0.000    0.000    0.050    0.010 precessing_surrogate.py:849(splinterp_many)
       53    0.000    0.000    0.050    0.001 surrogate.py:85(_splinterp_Cwrapper)
       53    0.006    0.000    0.049    0.001 spline_interp_Cwrapper.py:39(interpolate)
      262    0.021    0.000    0.021    0.000 {built-in method builtins.max}
      261    0.021    0.000    0.021    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:387(__call__)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:621(_integrate_forward)
      238    0.001    0.000    0.007    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:105(rotateWaveform)
        1    0.004    0.004    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:179(_eval_vector_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2390    0.001    0.000    0.004    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
     2390    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
     1257    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:556(_initial_RK4)
      212    0.000    0.000    0.000    0.000 _internal.py:280(data_as)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      212    0.000    0.000    0.000    0.000 __init__.py:613(cast)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      159    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:703(_assemble_mode_pair)
       58    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:827(rotate_spin)
      212    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         29367 function calls in 0.047 seconds

   Ordered by: cumulative time
   List reduced from 93 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.047    0.047 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.047    0.047 surrogate.py:1721(__call__)
        1    0.000    0.000    0.047    0.047 precessing_surrogate.py:933(__call__)
      196    0.000    0.000    0.026    0.000 surrogate.py:85(_splinterp_Cwrapper)
      196    0.005    0.000    0.025    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.021    0.004 precessing_surrogate.py:849(splinterp_many)
        1    0.000    0.000    0.014    0.014 precessing_surrogate.py:387(__call__)
      561    0.009    0.000    0.009    0.000 {built-in method builtins.max}
      560    0.009    0.000    0.009    0.000 {built-in method builtins.min}
      279    0.001    0.000    0.007    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:105(rotateWaveform)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
        1    0.004    0.004    0.005    0.005 precessing_surrogate.py:42(_wignerD_matrices)
      837    0.001    0.000    0.005    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:764(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:801(_eval_comp)
     2876    0.002    0.000    0.004    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:621(_integrate_forward)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:655(_integrate_backward)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
     2876    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      784    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      784    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1708    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:340(_get_t_from_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
      588    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
      784    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:703(_assemble_mode_pair)
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         45328 function calls in 0.107 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.107    0.107 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.107    0.107 surrogate.py:1721(__call__)
        1    0.000    0.000    0.106    0.106 precessing_surrogate.py:1263(__call__)
        1    0.001    0.001    0.106    0.106 precessing_surrogate.py:933(__call__)
      218    0.000    0.000    0.069    0.000 surrogate.py:85(_splinterp_Cwrapper)
      218    0.011    0.000    0.068    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.064    0.013 precessing_surrogate.py:849(splinterp_many)
      726    0.027    0.000    0.027    0.000 {built-in method builtins.max}
      725    0.027    0.000    0.027    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:387(__call__)
      546    0.001    0.000    0.018    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:621(_integrate_forward)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.012    0.012 precessing_surrogate.py:105(rotateWaveform)
     5135    0.003    0.000    0.012    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:655(_integrate_backward)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
     2610    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      872    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      872    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:340(_get_t_from_omega)
      654    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      224    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      872    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         45329 function calls in 0.087 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.087    0.087 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.087    0.087 surrogate.py:1721(__call__)
        1    0.000    0.000    0.086    0.086 precessing_surrogate.py:1263(__call__)
        1    0.000    0.000    0.086    0.086 precessing_surrogate.py:933(__call__)
      218    0.000    0.000    0.049    0.000 surrogate.py:85(_splinterp_Cwrapper)
      218    0.009    0.000    0.048    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.045    0.009 precessing_surrogate.py:849(splinterp_many)
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:387(__call__)
      726    0.018    0.000    0.018    0.000 {built-in method builtins.max}
      725    0.018    0.000    0.018    0.000 {built-in method builtins.min}
      546    0.001    0.000    0.018    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:621(_integrate_forward)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:105(rotateWaveform)
     5135    0.003    0.000    0.012    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:655(_integrate_backward)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
     2610    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      872    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      872    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:340(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
      654    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      224    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
      872    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         45328 function calls in 0.154 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.154    0.154 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.154    0.154 surrogate.py:1721(__call__)
        1    0.000    0.000    0.154    0.154 precessing_surrogate.py:1263(__call__)
        1    0.002    0.002    0.154    0.154 precessing_surrogate.py:933(__call__)
      218    0.000    0.000    0.115    0.001 surrogate.py:85(_splinterp_Cwrapper)
      218    0.015    0.000    0.114    0.001 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.111    0.022 precessing_surrogate.py:849(splinterp_many)
      726    0.048    0.000    0.048    0.000 {built-in method builtins.max}
      725    0.048    0.000    0.048    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:387(__call__)
      546    0.001    0.000    0.018    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:621(_integrate_forward)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:105(rotateWaveform)
     5135    0.003    0.000    0.012    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:655(_integrate_backward)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
     2610    0.002    0.000    0.002    0.000 {built-in method numpy.array}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
      872    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      872    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      654    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:340(_get_t_from_omega)
      224    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
       33    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         45329 function calls in 0.111 seconds

   Ordered by: cumulative time
   List reduced from 98 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.111    0.111 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.111    0.111 surrogate.py:1721(__call__)
        1    0.000    0.000    0.111    0.111 precessing_surrogate.py:1263(__call__)
        1    0.001    0.001    0.111    0.111 precessing_surrogate.py:933(__call__)
      218    0.000    0.000    0.073    0.000 surrogate.py:85(_splinterp_Cwrapper)
      218    0.011    0.000    0.073    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.069    0.014 precessing_surrogate.py:849(splinterp_many)
      726    0.029    0.000    0.029    0.000 {built-in method builtins.max}
      725    0.029    0.000    0.029    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:387(__call__)
      546    0.001    0.000    0.018    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:621(_integrate_forward)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.012    0.012 precessing_surrogate.py:105(rotateWaveform)
     5135    0.003    0.000    0.012    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:655(_integrate_backward)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
     2610    0.002    0.000    0.002    0.000 {built-in method numpy.array}
      872    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      872    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:340(_get_t_from_omega)
      654    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      224    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      872    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         36878 function calls in 0.388 seconds

   Ordered by: cumulative time
   List reduced from 86 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.388    0.388 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.388    0.388 surrogate.py:1721(__call__)
        1    0.000    0.000    0.388    0.388 precessing_surrogate.py:1263(__call__)
        1    0.008    0.008    0.388    0.388 precessing_surrogate.py:933(__call__)
        5    0.000    0.000    0.342    0.068 precessing_surrogate.py:849(splinterp_many)
       75    0.000    0.000    0.337    0.004 surrogate.py:85(_splinterp_Cwrapper)
       75    0.036    0.000    0.336    0.004 spline_interp_Cwrapper.py:39(interpolate)
      427    0.149    0.000    0.149    0.000 {built-in method builtins.max}
      426    0.147    0.000    0.147    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:387(__call__)
        1    0.001    0.001    0.018    0.018 precessing_surrogate.py:621(_integrate_forward)
      505    0.001    0.000    0.017    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
        1    0.000    0.000    0.016    0.016 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.002    0.002    0.016    0.016 precessing_surrogate.py:105(rotateWaveform)
        1    0.011    0.011    0.014    0.014 precessing_surrogate.py:42(_wignerD_matrices)
     1515    0.002    0.000    0.013    0.000 precessing_surrogate.py:179(_eval_vector_fit)
     4637    0.003    0.000    0.011    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
     2147    0.008    0.000    0.008    0.000 {built-in method numpy.array}
     4637    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
       81    0.003    0.000    0.003    0.000 {built-in method numpy.zeros}
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
      225    0.002    0.000    0.002    0.000 {method 'astype' of 'numpy.ndarray' objects}
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      300    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      300    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:556(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      300    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         46684 function calls in 0.122 seconds

   Ordered by: cumulative time
   List reduced from 96 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.122    0.122 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.122    0.122 surrogate.py:1721(__call__)
        1    0.000    0.000    0.122    0.122 precessing_surrogate.py:1263(__call__)
        1    0.001    0.001    0.122    0.122 precessing_surrogate.py:933(__call__)
      218    0.000    0.000    0.084    0.000 surrogate.py:85(_splinterp_Cwrapper)
      218    0.012    0.000    0.083    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.079    0.016 precessing_surrogate.py:849(splinterp_many)
      726    0.034    0.000    0.034    0.000 {built-in method builtins.max}
      725    0.034    0.000    0.034    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:387(__call__)
      546    0.001    0.000    0.018    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.012    0.012 precessing_surrogate.py:105(rotateWaveform)
     5274    0.003    0.000    0.012    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:655(_integrate_backward)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:621(_integrate_forward)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
     5274    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
     2749    0.002    0.000    0.002    0.000 {built-in method numpy.array}
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:340(_get_t_from_omega)
      872    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      872    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      654    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      224    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      872    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         36878 function calls in 0.117 seconds

   Ordered by: cumulative time
   List reduced from 86 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.117    0.117 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.117    0.117 surrogate.py:1721(__call__)
        1    0.000    0.000    0.117    0.117 precessing_surrogate.py:1263(__call__)
        1    0.001    0.001    0.117    0.117 precessing_surrogate.py:933(__call__)
        5    0.000    0.000    0.083    0.017 precessing_surrogate.py:849(splinterp_many)
       75    0.000    0.000    0.083    0.001 surrogate.py:85(_splinterp_Cwrapper)
       75    0.011    0.000    0.083    0.001 spline_interp_Cwrapper.py:39(interpolate)
      427    0.035    0.000    0.035    0.000 {built-in method builtins.max}
      426    0.035    0.000    0.035    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:387(__call__)
        1    0.001    0.001    0.018    0.018 precessing_surrogate.py:621(_integrate_forward)
      505    0.001    0.000    0.017    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
     1515    0.002    0.000    0.013    0.000 precessing_surrogate.py:179(_eval_vector_fit)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:842(inertial_waveform_modes)
        1    0.001    0.001    0.012    0.012 precessing_surrogate.py:105(rotateWaveform)
     4637    0.003    0.000    0.011    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
     2147    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      225    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      300    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
       81    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      300    0.000    0.000    0.000    0.000 __init__.py:613(cast)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:556(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      300    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         46684 function calls in 0.073 seconds

   Ordered by: cumulative time
   List reduced from 96 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.073    0.073 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.073    0.073 surrogate.py:1721(__call__)
        1    0.000    0.000    0.073    0.073 precessing_surrogate.py:1263(__call__)
        1    0.000    0.000    0.073    0.073 precessing_surrogate.py:933(__call__)
      218    0.000    0.000    0.036    0.000 surrogate.py:85(_splinterp_Cwrapper)
      218    0.007    0.000    0.035    0.000 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.031    0.006 precessing_surrogate.py:849(splinterp_many)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:387(__call__)
      546    0.001    0.000    0.018    0.000 precessing_surrogate.py:293(get_time_deriv_from_index)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:179(_eval_vector_fit)
      726    0.013    0.000    0.013    0.000 {built-in method builtins.max}
      725    0.013    0.000    0.013    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:842(inertial_waveform_modes)
     5274    0.003    0.000    0.012    0.000 precessing_surrogate.py:152(_eval_scalar_fit)
        1    0.001    0.001    0.012    0.012 precessing_surrogate.py:105(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:655(_integrate_backward)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:621(_integrate_forward)
       13    0.000    0.000    0.007    0.001 precessing_surrogate.py:311(get_time_deriv)
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:597(_one_backward_RK4_step)
     5274    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1189(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:764(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:801(_eval_comp)
      872    0.000    0.000    0.002    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:340(_get_t_from_omega)
     2749    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:334(get_omega)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      872    0.001    0.000    0.001    0.000 __init__.py:613(cast)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
      654    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:522(_initialize)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      872    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      224    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         68904 function calls (68884 primitive calls) in 0.481 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.482    0.482 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.002    0.002    0.482    0.482 surrogate.py:1721(__call__)
        1    0.000    0.000    0.473    0.473 surrogate.py:923(__call__)
        1    0.044    0.044    0.385    0.385 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.004    0.000    0.339    0.028 surrogate.py:85(_splinterp_Cwrapper)
       22    0.035    0.002    0.335    0.015 spline_interp_Cwrapper.py:39(interpolate)
       44    0.150    0.003    0.150    0.003 {built-in method builtins.min}
       44    0.146    0.003    0.146    0.003 {built-in method builtins.max}
       11    0.000    0.000    0.088    0.008 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.087    0.008 surrogate.py:401(__call__)
       20    0.000    0.000    0.087    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.075    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.075    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.069    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.069    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.044    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      158    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       21    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      224    0.004    0.000    0.004    0.000 {method 'astype' of 'numpy.ndarray' objects}
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      316    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         68900 function calls (68880 primitive calls) in 0.188 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.189    0.189 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.189    0.189 surrogate.py:1721(__call__)
        1    0.000    0.000    0.188    0.188 surrogate.py:923(__call__)
        1    0.012    0.012    0.100    0.100 surrogate.py:726(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.088    0.008 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.088    0.008 surrogate.py:401(__call__)
       20    0.000    0.000    0.088    0.004 surrogate.py:276(__call__)
    32/12    0.002    0.000    0.086    0.007 surrogate.py:85(_splinterp_Cwrapper)
       22    0.011    0.000    0.084    0.004 spline_interp_Cwrapper.py:39(interpolate)
      158    0.000    0.000    0.076    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.076    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.070    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.069    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.044    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
       44    0.035    0.001    0.035    0.001 {built-in method builtins.max}
       44    0.034    0.001    0.034    0.001 {built-in method builtins.min}
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      158    0.002    0.000    0.007    0.000 kernels.py:1525(__call__)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      158    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      224    0.004    0.000    0.004    0.000 {method 'astype' of 'numpy.ndarray' objects}
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      316    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         68904 function calls (68884 primitive calls) in 0.770 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.770    0.770 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.005    0.005    0.770    0.770 surrogate.py:1721(__call__)
        1    0.000    0.000    0.753    0.753 surrogate.py:923(__call__)
        1    0.073    0.073    0.665    0.665 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.014    0.000    0.589    0.049 surrogate.py:85(_splinterp_Cwrapper)
       22    0.057    0.003    0.575    0.026 spline_interp_Cwrapper.py:39(interpolate)
       44    0.258    0.006    0.258    0.006 {built-in method builtins.max}
       44    0.253    0.006    0.253    0.006 {built-in method builtins.min}
       11    0.000    0.000    0.088    0.008 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.088    0.008 surrogate.py:401(__call__)
       20    0.000    0.000    0.088    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.076    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.076    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.070    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.069    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.044    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
        1    0.000    0.000    0.009    0.009 {method 'update' of 'dict' objects}
       21    0.009    0.000    0.009    0.000 surrogate.py:2126(<genexpr>)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      224    0.005    0.000    0.005    0.000 {method 'astype' of 'numpy.ndarray' objects}
      158    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
        9    0.004    0.000    0.004    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         68900 function calls (68880 primitive calls) in 0.206 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.206    0.206 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.206    0.206 surrogate.py:1721(__call__)
        1    0.000    0.000    0.205    0.205 surrogate.py:923(__call__)
        1    0.014    0.014    0.117    0.117 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.002    0.000    0.102    0.009 surrogate.py:85(_splinterp_Cwrapper)
       22    0.013    0.001    0.100    0.005 spline_interp_Cwrapper.py:39(interpolate)
       11    0.000    0.000    0.088    0.008 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.088    0.008 surrogate.py:401(__call__)
       20    0.000    0.000    0.087    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.076    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.075    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.069    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.069    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.044    0.000 validation.py:2793(validate_data)
       44    0.042    0.001    0.042    0.001 {built-in method builtins.max}
       44    0.042    0.001    0.042    0.001 {built-in method builtins.min}
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      158    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      316    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      948    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         68877 function calls (68857 primitive calls) in 0.217 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.217    0.217 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.217    0.217 surrogate.py:1721(__call__)
        1    0.000    0.000    0.217    0.217 surrogate.py:923(__call__)
        1    0.016    0.016    0.129    0.129 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.001    0.000    0.111    0.009 surrogate.py:85(_splinterp_Cwrapper)
       22    0.016    0.001    0.110    0.005 spline_interp_Cwrapper.py:39(interpolate)
       11    0.000    0.000    0.088    0.008 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.088    0.008 surrogate.py:401(__call__)
       20    0.000    0.000    0.088    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.076    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.076    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.069    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.069    0.000 evaluate_fit.py:128(GPR_predict)
       44    0.046    0.001    0.046    0.001 {built-in method builtins.max}
       44    0.046    0.001    0.046    0.001 {built-in method builtins.min}
      316    0.002    0.000    0.044    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      158    0.000    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      316    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      948    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         68877 function calls (68857 primitive calls) in 3.666 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    3.667    3.667 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.034    0.034    3.667    3.667 surrogate.py:1721(__call__)
        1    0.000    0.000    3.611    3.611 surrogate.py:923(__call__)
        1    0.435    0.435    3.523    3.523 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.055    0.002    3.079    0.257 surrogate.py:85(_splinterp_Cwrapper)
       22    0.315    0.014    3.025    0.137 spline_interp_Cwrapper.py:39(interpolate)
       44    1.345    0.031    1.345    0.031 {built-in method builtins.min}
       44    1.319    0.030    1.319    0.030 {built-in method builtins.max}
       11    0.000    0.000    0.088    0.008 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.088    0.008 surrogate.py:401(__call__)
       20    0.000    0.000    0.088    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.076    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.076    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.070    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.069    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.045    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
      224    0.029    0.000    0.029    0.000 {method 'astype' of 'numpy.ndarray' objects}
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
        9    0.022    0.002    0.022    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      180    0.016    0.000    0.016    0.000 {built-in method numpy.zeros}
      316    0.004    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      158    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         68877 function calls (68857 primitive calls) in 0.173 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.174    0.174 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.174    0.174 surrogate.py:1721(__call__)
        1    0.000    0.000    0.173    0.173 surrogate.py:923(__call__)
       11    0.000    0.000    0.089    0.008 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.089    0.008 surrogate.py:401(__call__)
       20    0.000    0.000    0.089    0.004 surrogate.py:276(__call__)
        1    0.010    0.010    0.084    0.084 surrogate.py:726(_coorbital_to_inertial_frame)
      158    0.000    0.000    0.077    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.077    0.000 nodeFunction.py:110(__call__)
    32/12    0.002    0.000    0.072    0.006 surrogate.py:85(_splinterp_Cwrapper)
      158    0.000    0.000    0.071    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.071    0.000 evaluate_fit.py:247(gprfitEvaluator)
       22    0.010    0.000    0.070    0.003 spline_interp_Cwrapper.py:39(interpolate)
      158    0.001    0.000    0.070    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.045    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.041    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.037    0.000 validation.py:725(check_array)
       44    0.029    0.001    0.029    0.001 {built-in method builtins.max}
       44    0.029    0.001    0.029    0.001 {built-in method builtins.min}
      158    0.000    0.000    0.028    0.000 _base.py:297(predict)
      158    0.001    0.000    0.028    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.017    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      158    0.002    0.000    0.007    0.000 kernels.py:1525(__call__)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      158    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      316    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      948    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         68877 function calls (68857 primitive calls) in 0.866 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.867    0.867 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.006    0.006    0.867    0.867 surrogate.py:1721(__call__)
        1    0.000    0.000    0.856    0.856 surrogate.py:923(__call__)
        1    0.084    0.084    0.768    0.768 surrogate.py:726(_coorbital_to_inertial_frame)
    32/12    0.016    0.001    0.681    0.057 surrogate.py:85(_splinterp_Cwrapper)
       22    0.072    0.003    0.664    0.030 spline_interp_Cwrapper.py:39(interpolate)
       44    0.295    0.007    0.295    0.007 {built-in method builtins.min}
       44    0.291    0.007    0.291    0.007 {built-in method builtins.max}
       11    0.000    0.000    0.088    0.008 surrogate.py:409(_eval_sur)
       11    0.000    0.000    0.088    0.008 surrogate.py:401(__call__)
       20    0.000    0.000    0.088    0.004 surrogate.py:276(__call__)
      158    0.000    0.000    0.076    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.076    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.069    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.069    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.044    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      224    0.005    0.000    0.005    0.000 {method 'astype' of 'numpy.ndarray' objects}
      158    0.000    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
        9    0.004    0.000    0.004    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         22024 function calls (22016 primitive calls) in 0.170 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.170    0.170 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.001    0.001    0.170    0.170 surrogate.py:1721(__call__)
        1    0.000    0.000    0.167    0.167 surrogate.py:923(__call__)
        1    0.018    0.018    0.148    0.148 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.001    0.000    0.129    0.021 surrogate.py:85(_splinterp_Cwrapper)
       10    0.014    0.001    0.127    0.013 spline_interp_Cwrapper.py:39(interpolate)
       20    0.056    0.003    0.056    0.003 {built-in method builtins.max}
       20    0.056    0.003    0.056    0.003 {built-in method builtins.min}
        5    0.000    0.000    0.019    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.019    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.019    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
       14    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       80    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
        3    0.001    0.000    0.001    0.000 surrogate.py:716(_search_omega)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         22020 function calls (22012 primitive calls) in 0.043 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.043    0.043 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.043    0.043 surrogate.py:1721(__call__)
        1    0.000    0.000    0.043    0.043 surrogate.py:923(__call__)
        1    0.002    0.002    0.025    0.025 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.001    0.000    0.022    0.004 surrogate.py:85(_splinterp_Cwrapper)
       10    0.004    0.000    0.021    0.002 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.018    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.018    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.018    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
       20    0.008    0.000    0.008    0.000 {built-in method builtins.min}
       20    0.008    0.000    0.008    0.000 {built-in method builtins.max}
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.001    0.000 kernels.py:1525(__call__)
       80    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       50    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
      150    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         22024 function calls (22016 primitive calls) in 0.306 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.306    0.306 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.003    0.003    0.306    0.306 surrogate.py:1721(__call__)
        1    0.000    0.000    0.298    0.298 surrogate.py:923(__call__)
        1    0.034    0.034    0.279    0.279 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.003    0.000    0.243    0.040 surrogate.py:85(_splinterp_Cwrapper)
       10    0.024    0.002    0.240    0.024 spline_interp_Cwrapper.py:39(interpolate)
       20    0.107    0.005    0.107    0.005 {built-in method builtins.max}
       20    0.106    0.005    0.106    0.005 {built-in method builtins.min}
        5    0.000    0.000    0.019    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.019    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.019    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.016    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
        1    0.000    0.000    0.004    0.004 {method 'update' of 'dict' objects}
       11    0.004    0.000    0.004    0.000 surrogate.py:2126(<genexpr>)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       80    0.002    0.000    0.002    0.000 {method 'astype' of 'numpy.ndarray' objects}
       14    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       60    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
        3    0.001    0.000    0.001    0.000 surrogate.py:716(_search_omega)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         22020 function calls (22012 primitive calls) in 0.051 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.051    0.051 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.051    0.051 surrogate.py:1721(__call__)
        1    0.000    0.000    0.050    0.050 surrogate.py:923(__call__)
        1    0.003    0.003    0.032    0.032 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.001    0.000    0.029    0.005 surrogate.py:85(_splinterp_Cwrapper)
       10    0.004    0.000    0.028    0.003 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.018    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.018    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.018    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
       20    0.011    0.001    0.011    0.001 {built-in method builtins.max}
       20    0.011    0.001    0.011    0.001 {built-in method builtins.min}
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.001    0.000 kernels.py:1525(__call__)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       80    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
       50    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      100    0.000    0.000    0.001    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         22007 function calls (21999 primitive calls) in 0.054 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.054    0.054 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.054    0.054 surrogate.py:1721(__call__)
        1    0.000    0.000    0.054    0.054 surrogate.py:923(__call__)
        1    0.003    0.003    0.036    0.036 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.000    0.000    0.032    0.005 surrogate.py:85(_splinterp_Cwrapper)
       10    0.005    0.001    0.032    0.003 spline_interp_Cwrapper.py:39(interpolate)
        5    0.000    0.000    0.018    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.018    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.018    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
       20    0.013    0.001    0.013    0.001 {built-in method builtins.min}
       20    0.013    0.001    0.013    0.001 {built-in method builtins.max}
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       80    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
       50    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      150    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         22007 function calls (21999 primitive calls) in 1.613 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.614    1.614 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.021    0.021    1.614    1.614 surrogate.py:1721(__call__)
        1    0.000    0.000    1.580    1.580 surrogate.py:923(__call__)
        1    0.197    0.197    1.561    1.561 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.021    0.001    1.356    0.226 surrogate.py:85(_splinterp_Cwrapper)
       10    0.126    0.013    1.335    0.134 spline_interp_Cwrapper.py:39(interpolate)
       20    0.599    0.030    0.599    0.030 {built-in method builtins.max}
       20    0.590    0.030    0.590    0.030 {built-in method builtins.min}
        5    0.000    0.000    0.019    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.019    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.019    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.016    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.016    0.000 evaluate_fit.py:128(GPR_predict)
        5    0.012    0.002    0.012    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
       80    0.012    0.000    0.012    0.000 {method 'astype' of 'numpy.ndarray' objects}
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
       60    0.008    0.000    0.008    0.000 {built-in method numpy.zeros}
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       14    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.002    0.001    0.002    0.001 surrogate.py:716(_search_omega)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
        2    0.001    0.001    0.001    0.001 _function_base_impl.py:5577(append)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         22007 function calls (21999 primitive calls) in 0.038 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.038    0.038 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.038    0.038 surrogate.py:1721(__call__)
        1    0.000    0.000    0.038    0.038 surrogate.py:923(__call__)
        1    0.002    0.002    0.020    0.020 surrogate.py:726(_coorbital_to_inertial_frame)
        5    0.000    0.000    0.018    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.018    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.018    0.002 surrogate.py:276(__call__)
     14/6    0.000    0.000    0.018    0.003 surrogate.py:85(_splinterp_Cwrapper)
       10    0.003    0.000    0.017    0.002 spline_interp_Cwrapper.py:39(interpolate)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       20    0.007    0.000    0.007    0.000 {built-in method builtins.min}
       20    0.007    0.000    0.007    0.000 {built-in method builtins.max}
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       50    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
       80    0.001    0.000    0.001    0.000 {method 'astype' of 'numpy.ndarray' objects}
      150    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         22007 function calls (21999 primitive calls) in 0.347 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.348    0.348 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.003    0.003    0.348    0.348 surrogate.py:1721(__call__)
        1    0.000    0.000    0.343    0.343 surrogate.py:923(__call__)
        1    0.040    0.040    0.323    0.323 surrogate.py:726(_coorbital_to_inertial_frame)
     14/6    0.004    0.000    0.281    0.047 surrogate.py:85(_splinterp_Cwrapper)
       10    0.026    0.003    0.277    0.028 spline_interp_Cwrapper.py:39(interpolate)
       20    0.124    0.006    0.124    0.006 {built-in method builtins.max}
       20    0.123    0.006    0.123    0.006 {built-in method builtins.min}
        5    0.000    0.000    0.019    0.004 surrogate.py:409(_eval_sur)
        5    0.000    0.000    0.019    0.004 surrogate.py:401(__call__)
       10    0.000    0.000    0.019    0.002 surrogate.py:276(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.016    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.016    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
       80    0.003    0.000    0.003    0.000 {method 'astype' of 'numpy.ndarray' objects}
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
        5    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       14    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       60    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         81672 function calls (81654 primitive calls) in 0.446 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.446    0.446 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.002    0.002    0.446    0.446 surrogate.py:1721(__call__)
        1    0.000    0.000    0.440    0.440 surrogate.py:923(__call__)
        1    0.037    0.037    0.339    0.339 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.003    0.000    0.300    0.027 surrogate.py:85(_splinterp_Cwrapper)
       20    0.030    0.002    0.296    0.015 spline_interp_Cwrapper.py:39(interpolate)
       40    0.132    0.003    0.132    0.003 {built-in method builtins.min}
       40    0.130    0.003    0.130    0.003 {built-in method builtins.max}
       10    0.000    0.000    0.101    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.101    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.100    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.090    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.090    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.083    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.082    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.082    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.052    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.048    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.043    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.032    0.000 _base.py:297(predict)
      188    0.001    0.000    0.032    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
      376    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         81668 function calls (81650 primitive calls) in 0.191 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.192    0.192 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.192    0.192 surrogate.py:1721(__call__)
        1    0.000    0.000    0.191    0.191 surrogate.py:923(__call__)
       10    0.000    0.000    0.103    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.102    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.102    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.092    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.092    0.000 nodeFunction.py:110(__call__)
        1    0.010    0.010    0.088    0.088 surrogate.py:726(_coorbital_to_inertial_frame)
      188    0.000    0.000    0.084    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.084    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.084    0.000 evaluate_fit.py:128(GPR_predict)
    29/11    0.001    0.000    0.077    0.007 surrogate.py:85(_splinterp_Cwrapper)
       20    0.009    0.000    0.075    0.004 spline_interp_Cwrapper.py:39(interpolate)
      376    0.002    0.000    0.054    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.049    0.000 _gpr.py:373(predict)
      376    0.006    0.000    0.044    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.033    0.000 _base.py:297(predict)
      188    0.001    0.000    0.033    0.000 _base.py:287(_decision_function)
       40    0.033    0.001    0.033    0.001 {built-in method builtins.max}
       40    0.031    0.001    0.031    0.001 {built-in method builtins.min}
      188    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      376    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         81672 function calls (81654 primitive calls) in 0.709 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.710    0.710 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.003    0.003    0.710    0.710 surrogate.py:1721(__call__)
        1    0.000    0.000    0.696    0.696 surrogate.py:923(__call__)
        1    0.063    0.063    0.595    0.595 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.005    0.000    0.529    0.048 surrogate.py:85(_splinterp_Cwrapper)
       20    0.049    0.002    0.524    0.026 spline_interp_Cwrapper.py:39(interpolate)
       40    0.236    0.006    0.236    0.006 {built-in method builtins.max}
       40    0.232    0.006    0.232    0.006 {built-in method builtins.min}
       10    0.000    0.000    0.101    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.101    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.101    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.090    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.090    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.083    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.082    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.082    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.052    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.048    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.043    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.032    0.000 _base.py:297(predict)
      188    0.001    0.000    0.032    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
        1    0.000    0.000    0.007    0.007 {method 'update' of 'dict' objects}
       18    0.007    0.000    0.007    0.000 surrogate.py:2126(<genexpr>)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      248    0.004    0.000    0.004    0.000 {method 'astype' of 'numpy.ndarray' objects}
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         81668 function calls (81650 primitive calls) in 0.209 seconds

   Ordered by: cumulative time
   List reduced from 173 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.210    0.210 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.210    0.210 surrogate.py:1721(__call__)
        1    0.000    0.000    0.209    0.209 surrogate.py:923(__call__)
        1    0.012    0.012    0.108    0.108 surrogate.py:726(_coorbital_to_inertial_frame)
       10    0.000    0.000    0.100    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.100    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.100    0.006 surrogate.py:276(__call__)
    29/11    0.001    0.000    0.094    0.009 surrogate.py:85(_splinterp_Cwrapper)
       20    0.011    0.001    0.093    0.005 spline_interp_Cwrapper.py:39(interpolate)
      188    0.000    0.000    0.090    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.090    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.083    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.082    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.082    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.052    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.048    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.043    0.000 validation.py:725(check_array)
       40    0.040    0.001    0.040    0.001 {built-in method builtins.max}
       40    0.039    0.001    0.039    0.001 {built-in method builtins.min}
      188    0.000    0.000    0.032    0.000 _base.py:297(predict)
      188    0.001    0.000    0.032    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      376    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         81648 function calls (81630 primitive calls) in 0.219 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.220    0.220 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.220    0.220 surrogate.py:1721(__call__)
        1    0.000    0.000    0.219    0.219 surrogate.py:923(__call__)
        1    0.014    0.014    0.119    0.119 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.001    0.000    0.104    0.009 surrogate.py:85(_splinterp_Cwrapper)
       20    0.014    0.001    0.103    0.005 spline_interp_Cwrapper.py:39(interpolate)
       10    0.000    0.000    0.100    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.100    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.100    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.089    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.089    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.082    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.082    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.081    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.052    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.048    0.000 _gpr.py:373(predict)
       40    0.044    0.001    0.044    0.001 {built-in method builtins.max}
       40    0.043    0.001    0.043    0.001 {built-in method builtins.min}
      376    0.005    0.000    0.042    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.032    0.000 _base.py:297(predict)
      188    0.001    0.000    0.032    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      376    0.000    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         81648 function calls (81630 primitive calls) in 3.298 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    3.299    3.299 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.031    0.031    3.299    3.299 surrogate.py:1721(__call__)
        1    0.000    0.000    3.249    3.249 surrogate.py:923(__call__)
        1    0.375    0.375    3.149    3.149 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.049    0.002    2.766    0.251 surrogate.py:85(_splinterp_Cwrapper)
       20    0.255    0.013    2.716    0.136 spline_interp_Cwrapper.py:39(interpolate)
       40    1.224    0.031    1.224    0.031 {built-in method builtins.min}
       40    1.197    0.030    1.197    0.030 {built-in method builtins.max}
       10    0.000    0.000    0.100    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.100    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.100    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.089    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.089    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.082    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.082    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.081    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.052    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.048    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.043    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.032    0.000 _base.py:297(predict)
      188    0.001    0.000    0.032    0.000 _base.py:287(_decision_function)
      248    0.025    0.000    0.025    0.000 {method 'astype' of 'numpy.ndarray' objects}
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
        7    0.019    0.003    0.019    0.003 {method 'conjugate' of 'numpy.ndarray' objects}
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      208    0.014    0.000    0.014    0.000 {built-in method numpy.zeros}
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
        4    0.004    0.001    0.004    0.001 _function_base_impl.py:1402(diff)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         81648 function calls (81630 primitive calls) in 0.177 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.178    0.178 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.178    0.178 surrogate.py:1721(__call__)
        1    0.000    0.000    0.178    0.178 surrogate.py:923(__call__)
       10    0.000    0.000    0.102    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.102    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.102    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.092    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.092    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.084    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.084    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.084    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.009    0.009    0.075    0.075 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.001    0.000    0.065    0.006 surrogate.py:85(_splinterp_Cwrapper)
       20    0.008    0.000    0.063    0.003 spline_interp_Cwrapper.py:39(interpolate)
      376    0.002    0.000    0.054    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.048    0.000 _gpr.py:373(predict)
      376    0.006    0.000    0.044    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.033    0.000 _base.py:297(predict)
      188    0.001    0.000    0.033    0.000 _base.py:287(_decision_function)
       40    0.027    0.001    0.027    0.001 {built-in method builtins.max}
       40    0.026    0.001    0.026    0.001 {built-in method builtins.min}
      188    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      564    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      188    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.005    0.000 base.py:1166(__sklearn_tags__)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      376    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         81648 function calls (81630 primitive calls) in 0.797 seconds

   Ordered by: cumulative time
   List reduced from 171 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.798    0.798 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.008    0.008    0.798    0.798 surrogate.py:1721(__call__)
        1    0.000    0.000    0.787    0.787 surrogate.py:923(__call__)
        1    0.070    0.070    0.687    0.687 surrogate.py:726(_coorbital_to_inertial_frame)
    29/11    0.013    0.000    0.614    0.056 surrogate.py:85(_splinterp_Cwrapper)
       20    0.056    0.003    0.601    0.030 spline_interp_Cwrapper.py:39(interpolate)
       40    0.271    0.007    0.271    0.007 {built-in method builtins.min}
       40    0.267    0.007    0.267    0.007 {built-in method builtins.max}
       10    0.000    0.000    0.100    0.010 surrogate.py:409(_eval_sur)
       10    0.000    0.000    0.100    0.010 surrogate.py:401(__call__)
       17    0.000    0.000    0.100    0.006 surrogate.py:276(__call__)
      188    0.000    0.000    0.090    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.089    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.082    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.082    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.081    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.052    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.048    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.043    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.032    0.000 _base.py:297(predict)
      188    0.001    0.000    0.032    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      248    0.005    0.000    0.005    0.000 {method 'astype' of 'numpy.ndarray' objects}
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
      376    0.000    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
```

#### PR-70

##### NRSur7dq4 / mks_dt_0.000244140625_flow_0

```text
         27100 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:626(_integrate_forward)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      837    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.004    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
     2806    0.002    0.000    0.004    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.002    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
     2806    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1633    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
```

##### NRSur7dq4 / mks_dt_0.000244140625_flow_20

```text
         27101 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.008    0.008 precessing_surrogate.py:847(inertial_waveform_modes)
      279    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:626(_integrate_forward)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      837    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2806    0.002    0.000    0.004    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        4    0.000    0.000    0.002    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
     2806    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:660(_integrate_backward)
     1633    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_0

```text
         27100 function calls in 0.031 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.031    0.031 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.031    0.031 surrogate.py:1721(__call__)
        1    0.000    0.000    0.031    0.031 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:110(rotateWaveform)
        1    0.006    0.006    0.008    0.008 precessing_surrogate.py:42(_wignerD_matrices)
      279    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:626(_integrate_forward)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      837    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.004    0.000 precessing_surrogate.py:806(_eval_comp)
     2806    0.002    0.000    0.004    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
     2806    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
     1633    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
       22    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
```

##### NRSur7dq4 / mks_dt_0.0001220703125_flow_20

```text
         27101 function calls in 0.030 seconds

   Ordered by: cumulative time
   List reduced from 103 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.030    0.030 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.030    0.030 surrogate.py:1721(__call__)
        1    0.000    0.000    0.030    0.030 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.009    0.009 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:110(rotateWaveform)
        1    0.006    0.006    0.008    0.008 precessing_surrogate.py:42(_wignerD_matrices)
      279    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:626(_integrate_forward)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
      837    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
     2806    0.002    0.000    0.004    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        4    0.000    0.000    0.003    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     2806    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      574    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.002    0.000 precessing_surrogate.py:39(_assemble_powers)
     1633    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:660(_integrate_backward)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      574    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2806    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:345(_get_t_from_omega)
       47    0.000    0.000    0.000    0.000 precessing_surrogate.py:339(get_omega)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
```

##### NRSur7dq4 / geom_dt_0.1_flow_0

```text
         20326 function calls in 0.026 seconds

   Ordered by: cumulative time
   List reduced from 91 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.026    0.026 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.026    0.026 surrogate.py:1721(__call__)
        1    0.000    0.000    0.026    0.026 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
      238    0.001    0.000    0.006    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        4    0.000    0.000    0.006    0.002 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.006    0.006 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.006    0.006    0.006    0.006 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      714    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     2390    0.001    0.000    0.003    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
     2390    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
     1252    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
        5    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
```

##### NRSur7dq4 / geom_dt_0.1_flow_0.01

```text
         27707 function calls in 0.028 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.028    0.028 surrogate.py:1721(__call__)
        1    0.000    0.000    0.028    0.028 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:110(rotateWaveform)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.004    0.004    0.005    0.005 precessing_surrogate.py:42(_wignerD_matrices)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      837    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2876    0.002    0.000    0.004    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:660(_integrate_backward)
        4    0.000    0.000    0.002    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
      644    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
     2876    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1703    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0

```text
         20326 function calls in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 91 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.022    0.022 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.022    0.022 surrogate.py:1721(__call__)
        1    0.000    0.000    0.022    0.022 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:392(__call__)
        1    0.000    0.000    0.007    0.007 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:110(rotateWaveform)
        1    0.001    0.001    0.007    0.007 precessing_surrogate.py:626(_integrate_forward)
      238    0.001    0.000    0.006    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.005    0.005    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
      714    0.001    0.000    0.004    0.000 precessing_surrogate.py:184(_eval_vector_fit)
     2390    0.001    0.000    0.003    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        4    0.000    0.000    0.002    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
      498    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     2390    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
      486    0.001    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      498    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
     1252    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
      486    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2158    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2390    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      748    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
       44    0.000    0.000    0.000    0.000 {method 'dot' of 'numpy.ndarray' objects}
      996    0.000    0.000    0.000    0.000 {built-in method numpy.asanyarray}
      568    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
      238    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
       74    0.000    0.000    0.000    0.000 __init__.py:613(cast)
      226    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
       22    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
```

##### NRSur7dq4 / geom_dt_0.5_flow_0.01

```text
         27707 function calls in 0.027 seconds

   Ordered by: cumulative time
   List reduced from 101 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.027    0.027 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.027    0.027 surrogate.py:1721(__call__)
        1    0.000    0.000    0.027    0.027 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:392(__call__)
      279    0.001    0.000    0.007    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.006    0.006 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.006    0.006 precessing_surrogate.py:110(rotateWaveform)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        1    0.004    0.004    0.006    0.006 precessing_surrogate.py:42(_wignerD_matrices)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      837    0.001    0.000    0.005    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.005    0.005 precessing_surrogate.py:769(__call__)
       42    0.001    0.000    0.005    0.000 precessing_surrogate.py:806(_eval_comp)
     2876    0.002    0.000    0.004    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.003    0.003 precessing_surrogate.py:660(_integrate_backward)
      644    0.002    0.000    0.002    0.000 surrogate.py:2446(get_fit_params)
      503    0.001    0.000    0.002    0.000 _function_base_impl.py:5577(append)
     2876    0.002    0.000    0.002    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.002    0.000 surrogate.py:105(_splinterp_Cwrapper_many)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
        1    0.000    0.000    0.001    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      646    0.001    0.000    0.001    0.000 __init__.py:613(cast)
     1703    0.001    0.000    0.001    0.000 {built-in method numpy.array}
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      503    0.000    0.000    0.001    0.000 fromnumeric.py:1879(ravel)
      117    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      644    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     2480    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
     2876    0.000    0.000    0.000    0.000 surrogate.py:2472(get_fit_settings)
      646    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      753    0.000    0.000    0.000    0.000 {method 'ravel' of 'numpy.ndarray' objects}
        9    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_0

```text
         43096 function calls in 0.043 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.043    0.043 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.043    0.043 surrogate.py:1721(__call__)
        1    0.000    0.000    0.043    0.043 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.043    0.043 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.017    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.012    0.012 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.011    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
     5135    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
      144    0.002    0.000    0.004    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        4    0.000    0.000    0.003    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
```

##### SEOBNRv4PHMSur / mks_dt_0.000244140625_flow_20

```text
         43097 function calls in 0.042 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.042    0.042 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.042    0.042 surrogate.py:1721(__call__)
        1    0.000    0.000    0.042    0.042 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.042    0.042 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.017    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.012    0.012 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.011    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
     5135    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        4    0.000    0.000    0.002    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.002    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_0

```text
         43096 function calls in 0.045 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.045    0.045 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.045    0.045 surrogate.py:1721(__call__)
        1    0.000    0.000    0.044    0.044 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.044    0.044 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.024    0.024 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.017    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.012    0.012 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.011    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     5135    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        4    0.000    0.000    0.004    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
        1    0.000    0.000    0.004    0.004 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.004    0.004 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        1    0.000    0.000    0.001    0.001 {method 'update' of 'dict' objects}
       33    0.001    0.000    0.001    0.000 surrogate.py:2126(<genexpr>)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
```

##### SEOBNRv4PHMSur / mks_dt_0.0001220703125_flow_20

```text
         43097 function calls in 0.044 seconds

   Ordered by: cumulative time
   List reduced from 106 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.044    0.044 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.044    0.044 surrogate.py:1721(__call__)
        1    0.000    0.000    0.043    0.043 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.043    0.043 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.025    0.025 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.018    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.001    0.001    0.014    0.014 precessing_surrogate.py:626(_integrate_forward)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     5135    0.003    0.000    0.012    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     5135    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
        1    0.000    0.000    0.004    0.004 precessing_surrogate.py:660(_integrate_backward)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5135    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        4    0.000    0.000    0.003    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.002    0.002    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     2605    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     5135    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
      767    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      129    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      767    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}
       33    0.000    0.000    0.000    0.000 surrogate.py:2126(<genexpr>)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0

```text
         35075 function calls in 0.044 seconds

   Ordered by: cumulative time
   List reduced from 94 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.044    0.044 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.044    0.044 surrogate.py:1721(__call__)
        1    0.000    0.000    0.044    0.044 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.044    0.044 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.018    0.018 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.016    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1515    0.002    0.000    0.013    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.012    0.012 precessing_surrogate.py:110(rotateWaveform)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
        4    0.000    0.000    0.011    0.003 surrogate.py:105(_splinterp_Cwrapper_many)
     4637    0.002    0.000    0.011    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.000    0.000    0.010    0.010 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.010    0.010    0.010    0.010 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     2142    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
       33    0.000    0.000    0.000    0.000 spline_interp_Cwrapper.py:153(<genexpr>)
```

##### SEOBNRv4PHMSur / geom_dt_0.1_flow_0.01

```text
         44452 function calls in 0.044 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.044    0.044 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.044    0.044 surrogate.py:1721(__call__)
        1    0.000    0.000    0.044    0.044 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.044    0.044 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.025    0.025 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.018    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.012    0.012 precessing_surrogate.py:110(rotateWaveform)
     5274    0.003    0.000    0.012    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:660(_integrate_backward)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:626(_integrate_forward)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     5274    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.005    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
        4    0.000    0.000    0.004    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.003    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:345(_get_t_from_omega)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
     2744    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
      814    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0

```text
         35075 function calls in 0.037 seconds

   Ordered by: cumulative time
   List reduced from 94 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.037    0.037 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.037    0.037 surrogate.py:1721(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.037    0.037 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.018    0.018 precessing_surrogate.py:392(__call__)
        1    0.001    0.001    0.017    0.017 precessing_surrogate.py:626(_integrate_forward)
      505    0.001    0.000    0.016    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
        1    0.000    0.000    0.013    0.013 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.013    0.013 precessing_surrogate.py:110(rotateWaveform)
     1515    0.003    0.000    0.013    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.010    0.010    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
     4637    0.002    0.000    0.011    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
     4637    0.005    0.000    0.005    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        4    0.000    0.000    0.004    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
        1    0.000    0.000    0.003    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.003    0.003    0.003    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     4637    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
     4637    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
     2142    0.001    0.000    0.001    0.000 {built-in method numpy.array}
      597    0.000    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      186    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
     4138    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:561(_initial_RK4)
      597    0.000    0.000    0.000    0.000 _function_base_impl.py:924(copy)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
      186    0.000    0.000    0.000    0.000 fromnumeric.py:1879(ravel)
        6    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
      493    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.ab4_dy}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.assemble_dydt}
      505    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.get_ds_fit_x}
      496    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.normalize_y}
        4    0.000    0.000    0.000    0.000 precessing_surrogate.py:832(rotate_spin)
        5    0.000    0.000    0.000    0.000 precessing_surrogate.py:708(_assemble_mode_pair)
        1    0.000    0.000    0.000    0.000 precessing_surrogate.py:842(coorb_spins_from_copr_spins)
        2    0.000    0.000    0.000    0.000 precessing_surrogate.py:22(multiplyQuats)
```

##### SEOBNRv4PHMSur / geom_dt_0.5_flow_0.01

```text
         44452 function calls in 0.043 seconds

   Ordered by: cumulative time
   List reduced from 104 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.043    0.043 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.043    0.043 surrogate.py:1721(__call__)
        1    0.000    0.000    0.043    0.043 precessing_surrogate.py:1265(__call__)
        1    0.000    0.000    0.043    0.043 precessing_surrogate.py:937(__call__)
        1    0.000    0.000    0.025    0.025 precessing_surrogate.py:392(__call__)
      546    0.001    0.000    0.018    0.000 precessing_surrogate.py:298(get_time_deriv_from_index)
     1638    0.003    0.000    0.014    0.000 precessing_surrogate.py:184(_eval_vector_fit)
        1    0.000    0.000    0.012    0.012 precessing_surrogate.py:847(inertial_waveform_modes)
        1    0.001    0.001    0.012    0.012 precessing_surrogate.py:110(rotateWaveform)
     5274    0.003    0.000    0.012    0.000 precessing_surrogate.py:157(_eval_scalar_fit)
        1    0.009    0.009    0.011    0.011 precessing_surrogate.py:42(_wignerD_matrices)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:660(_integrate_backward)
        1    0.001    0.001    0.009    0.009 precessing_surrogate.py:626(_integrate_forward)
       13    0.000    0.000    0.006    0.000 precessing_surrogate.py:316(get_time_deriv)
     5274    0.006    0.000    0.006    0.000 {built-in method gwsurrogate.precessing_utils._utils.eval_fit}
        3    0.000    0.000    0.006    0.002 precessing_surrogate.py:602(_one_backward_RK4_step)
      144    0.000    0.000    0.004    0.000 surrogate.py:90(_splinterp_Cwrapper)
      144    0.002    0.000    0.003    0.000 spline_interp_Cwrapper.py:50(interpolate)
     5274    0.002    0.000    0.003    0.000 precessing_surrogate.py:1191(<lambda>)
        1    0.000    0.000    0.002    0.002 precessing_surrogate.py:769(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:105(_splinterp_Cwrapper_many)
       20    0.000    0.000    0.002    0.000 precessing_surrogate.py:806(_eval_comp)
        1    0.000    0.000    0.002    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:345(_get_t_from_omega)
        1    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      906    0.001    0.000    0.001    0.000 surrogate.py:2578(get_fit_params)
      268    0.000    0.000    0.001    0.000 precessing_surrogate.py:339(get_omega)
     2744    0.001    0.000    0.001    0.000 {built-in method numpy.array}
     5274    0.001    0.000    0.001    0.000 surrogate.py:2588(get_fit_settings)
        4    0.001    0.000    0.001    0.000 precessing_surrogate.py:39(_assemble_powers)
      584    0.000    0.000    0.001    0.000 _internal.py:280(data_as)
      668    0.001    0.000    0.001    0.000 __init__.py:613(cast)
      191    0.000    0.000    0.001    0.000 _function_base_impl.py:5577(append)
     4460    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
      906    0.000    0.000    0.001    0.000 _function_base_impl.py:924(copy)
        3    0.000    0.000    0.001    0.000 precessing_surrogate.py:854(splinterp_many)
        1    0.000    0.000    0.001    0.001 precessing_surrogate.py:527(_initialize)
        3    0.000    0.000    0.001    0.000 spline_interp_Cwrapper.py:67(interpolate_many)
      668    0.000    0.000    0.000    0.000 _internal.py:263(__init__)
     1140    0.000    0.000    0.000    0.000 {built-in method gwsurrogate.precessing_utils._utils.binom}
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_7

```text
         68518 function calls in 0.199 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.200    0.200 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.002    0.002    0.200    0.200 surrogate.py:1721(__call__)
        1    0.000    0.000    0.193    0.193 surrogate.py:938(__call__)
        1    0.071    0.071    0.098    0.098 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.095    0.009 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.095    0.009 surrogate.py:416(__call__)
       20    0.000    0.000    0.095    0.005 surrogate.py:291(__call__)
      158    0.000    0.000    0.075    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.075    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.069    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.068    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.044    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
      316    0.004    0.000    0.036    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
       12    0.000    0.000    0.023    0.002 surrogate.py:90(_splinterp_Cwrapper)
       24    0.019    0.001    0.019    0.001 {method 'dot' of 'numpy.ndarray' objects}
       10    0.000    0.000    0.017    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.016    0.002    0.017    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
        2    0.006    0.003    0.006    0.003 spline_interp_Cwrapper.py:50(interpolate)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      158    0.000    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
        1    0.000    0.000    0.003    0.003 {method 'update' of 'dict' objects}
       21    0.003    0.000    0.003    0.000 surrogate.py:2126(<genexpr>)
      316    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur3dq8 / mks_dt_0.000244140625_flow_20

```text
         68514 function calls in 0.111 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.111    0.111 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.111    0.111 surrogate.py:1721(__call__)
        1    0.000    0.000    0.111    0.111 surrogate.py:938(__call__)
       11    0.000    0.000    0.090    0.008 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.090    0.008 surrogate.py:416(__call__)
       20    0.000    0.000    0.089    0.004 surrogate.py:291(__call__)
      158    0.000    0.000    0.078    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.077    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.071    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.071    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.071    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.045    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.041    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.037    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.028    0.000 _base.py:297(predict)
      158    0.001    0.000    0.028    0.000 _base.py:287(_decision_function)
        1    0.012    0.012    0.021    0.021 surrogate.py:741(_coorbital_to_inertial_frame)
      158    0.001    0.000    0.017    0.000 kernels.py:833(__call__)
      316    0.004    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       12    0.000    0.000    0.007    0.001 surrogate.py:90(_splinterp_Cwrapper)
      158    0.002    0.000    0.007    0.000 kernels.py:1525(__call__)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
       10    0.000    0.000    0.005    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.004    0.000    0.005    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      158    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      316    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      948    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
     1264    0.001    0.000    0.003    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_7

```text
         68518 function calls in 0.252 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.252    0.252 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.005    0.005    0.252    0.252 surrogate.py:1721(__call__)
        1    0.000    0.000    0.236    0.236 surrogate.py:938(__call__)
        1    0.102    0.102    0.148    0.148 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.087    0.008 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.087    0.008 surrogate.py:416(__call__)
       20    0.000    0.000    0.087    0.004 surrogate.py:291(__call__)
      158    0.000    0.000    0.075    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.075    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.069    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.069    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.044    0.000 validation.py:2793(validate_data)
       12    0.000    0.000    0.042    0.004 surrogate.py:90(_splinterp_Cwrapper)
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
       10    0.000    0.000    0.031    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.030    0.003    0.031    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
        2    0.011    0.005    0.011    0.006 spline_interp_Cwrapper.py:50(interpolate)
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
        1    0.000    0.000    0.008    0.008 {method 'update' of 'dict' objects}
       21    0.008    0.000    0.008    0.000 surrogate.py:2126(<genexpr>)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      158    0.000    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
        9    0.003    0.000    0.003    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
```

##### NRHybSur3dq8 / mks_dt_0.0001220703125_flow_20

```text
         68514 function calls in 0.114 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.115    0.115 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.115    0.115 surrogate.py:1721(__call__)
        1    0.000    0.000    0.114    0.114 surrogate.py:938(__call__)
       11    0.000    0.000    0.089    0.008 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.089    0.008 surrogate.py:416(__call__)
       20    0.000    0.000    0.088    0.004 surrogate.py:291(__call__)
      158    0.000    0.000    0.077    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.076    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.070    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.070    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.070    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.045    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.041    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
        1    0.015    0.015    0.024    0.024 surrogate.py:741(_coorbital_to_inertial_frame)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.004    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       12    0.000    0.000    0.008    0.001 surrogate.py:90(_splinterp_Cwrapper)
      158    0.002    0.000    0.007    0.000 kernels.py:1525(__call__)
       10    0.000    0.000    0.006    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.006    0.001    0.006    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      158    0.000    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      316    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
      948    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      158    0.001    0.000    0.003    0.000 _py_warnings.py:254(filterwarnings)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.01

```text
         68491 function calls in 0.116 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.116    0.116 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.116    0.116 surrogate.py:1721(__call__)
        1    0.000    0.000    0.116    0.116 surrogate.py:938(__call__)
       11    0.000    0.000    0.088    0.008 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.088    0.008 surrogate.py:416(__call__)
       20    0.000    0.000    0.088    0.004 surrogate.py:291(__call__)
      158    0.000    0.000    0.077    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.076    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.070    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.070    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.070    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.045    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.041    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
        1    0.017    0.017    0.027    0.027 surrogate.py:741(_coorbital_to_inertial_frame)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
       12    0.000    0.000    0.008    0.001 surrogate.py:90(_splinterp_Cwrapper)
      158    0.002    0.000    0.007    0.000 kernels.py:1525(__call__)
       10    0.000    0.000    0.006    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.006    0.001    0.006    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      158    0.000    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      316    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      948    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     1264    0.001    0.000    0.003    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8 / geom_dt_0.1_flow_0.002

```text
         68491 function calls in 0.777 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.777    0.777 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.035    0.035    0.777    0.777 surrogate.py:1721(__call__)
        1    0.000    0.000    0.721    0.721 surrogate.py:938(__call__)
        1    0.437    0.437    0.633    0.633 surrogate.py:741(_coorbital_to_inertial_frame)
       12    0.000    0.000    0.187    0.016 surrogate.py:90(_splinterp_Cwrapper)
       10    0.000    0.000    0.132    0.013 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.131    0.013    0.132    0.013 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       11    0.000    0.000    0.088    0.008 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.088    0.008 surrogate.py:416(__call__)
       20    0.000    0.000    0.087    0.004 surrogate.py:291(__call__)
      158    0.000    0.000    0.075    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.075    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.069    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.068    0.000 evaluate_fit.py:128(GPR_predict)
        2    0.055    0.027    0.055    0.027 spline_interp_Cwrapper.py:50(interpolate)
      316    0.002    0.000    0.044    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
        9    0.021    0.002    0.021    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.012    0.000    0.012    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
        4    0.005    0.001    0.005    0.001 _function_base_impl.py:1402(diff)
      158    0.000    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      316    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.01

```text
         68491 function calls in 0.107 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.108    0.108 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.108    0.108 surrogate.py:1721(__call__)
        1    0.000    0.000    0.107    0.107 surrogate.py:938(__call__)
       11    0.000    0.000    0.088    0.008 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.088    0.008 surrogate.py:416(__call__)
       20    0.000    0.000    0.088    0.004 surrogate.py:291(__call__)
      158    0.000    0.000    0.077    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.076    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.070    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.070    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.070    0.000 evaluate_fit.py:128(GPR_predict)
      316    0.002    0.000    0.045    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.041    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
        1    0.011    0.011    0.019    0.019 surrogate.py:741(_coorbital_to_inertial_frame)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.004    0.000    0.014    0.000 validation.py:103(_assert_all_finite)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      158    0.002    0.000    0.007    0.000 kernels.py:1525(__call__)
       12    0.000    0.000    0.006    0.001 surrogate.py:90(_splinterp_Cwrapper)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
       10    0.000    0.000    0.005    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.004    0.000    0.005    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      158    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      948    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      316    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
      474    0.001    0.000    0.003    0.000 base.py:603(__sklearn_tags__)
     1264    0.001    0.000    0.003    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8 / geom_dt_0.5_flow_0.002

```text
         68491 function calls in 0.263 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.263    0.263 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.006    0.006    0.263    0.263 surrogate.py:1721(__call__)
        1    0.000    0.000    0.253    0.253 surrogate.py:938(__call__)
        1    0.116    0.116    0.165    0.165 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.087    0.008 surrogate.py:424(_eval_sur)
       11    0.000    0.000    0.087    0.008 surrogate.py:416(__call__)
       20    0.000    0.000    0.087    0.004 surrogate.py:291(__call__)
      158    0.000    0.000    0.075    0.000 nodeFunction.py:205(__call__)
      158    0.002    0.000    0.075    0.000 nodeFunction.py:110(__call__)
      158    0.000    0.000    0.069    0.000 nodeFunction.py:96(__call__)
      158    0.000    0.000    0.069    0.000 evaluate_fit.py:247(gprfitEvaluator)
      158    0.001    0.000    0.068    0.000 evaluate_fit.py:128(GPR_predict)
       12    0.000    0.000    0.046    0.004 surrogate.py:90(_splinterp_Cwrapper)
      316    0.002    0.000    0.044    0.000 validation.py:2793(validate_data)
      158    0.002    0.000    0.040    0.000 _gpr.py:373(predict)
      316    0.005    0.000    0.036    0.000 validation.py:725(check_array)
       10    0.000    0.000    0.033    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
       10    0.032    0.003    0.033    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      158    0.000    0.000    0.027    0.000 _base.py:297(predict)
      158    0.001    0.000    0.027    0.000 _base.py:287(_decision_function)
      158    0.001    0.000    0.016    0.000 kernels.py:833(__call__)
      316    0.003    0.000    0.013    0.000 validation.py:103(_assert_all_finite)
        2    0.013    0.006    0.013    0.006 spline_interp_Cwrapper.py:50(interpolate)
      158    0.001    0.000    0.012    0.000 kernels.py:931(__call__)
       24    0.011    0.000    0.011    0.000 {method 'dot' of 'numpy.ndarray' objects}
      948    0.004    0.000    0.011    0.000 validation.py:371(_num_samples)
      158    0.002    0.000    0.006    0.000 kernels.py:1525(__call__)
     6164    0.003    0.000    0.005    0.000 {built-in method builtins.isinstance}
      158    0.000    0.000    0.005    0.000 kernels.py:1239(__call__)
      474    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      316    0.001    0.000    0.004    0.000 _array_api.py:857(_asarray_with_order)
      474    0.002    0.000    0.004    0.000 _py_warnings.py:320(_add_filter)
      158    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      158    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
        9    0.004    0.000    0.004    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
      474    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
     9322    0.003    0.000    0.003    0.000 {built-in method builtins.hasattr}
      632    0.001    0.000    0.003    0.000 _aliases.py:89(asarray)
      948    0.001    0.000    0.003    0.000 _array_api.py:331(get_namespace)
      316    0.000    0.000    0.003    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_7

```text
         21866 function calls in 0.051 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.051    0.051 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.001    0.001    0.051    0.051 surrogate.py:1721(__call__)
        1    0.000    0.000    0.048    0.048 surrogate.py:938(__call__)
        1    0.018    0.018    0.029    0.029 surrogate.py:741(_coorbital_to_inertial_frame)
        5    0.000    0.000    0.019    0.004 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.019    0.004 surrogate.py:416(__call__)
       10    0.000    0.000    0.019    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.016    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.016    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
        6    0.000    0.000    0.009    0.002 surrogate.py:90(_splinterp_Cwrapper)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
        4    0.000    0.000    0.005    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.005    0.001    0.005    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.004    0.002    0.004    0.002 spline_interp_Cwrapper.py:50(interpolate)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
        1    0.000    0.000    0.002    0.002 {method 'update' of 'dict' objects}
       11    0.002    0.000    0.002    0.000 surrogate.py:2126(<genexpr>)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
```

##### NRHybSur2dq15 / mks_dt_0.000244140625_flow_20

```text
         21862 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 surrogate.py:938(__call__)
        5    0.000    0.000    0.018    0.004 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.018    0.004 surrogate.py:416(__call__)
       10    0.000    0.000    0.018    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
        1    0.002    0.002    0.006    0.006 surrogate.py:741(_coorbital_to_inertial_frame)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
        6    0.000    0.000    0.003    0.000 surrogate.py:90(_splinterp_Cwrapper)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
        4    0.000    0.000    0.002    0.000 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.001    0.000    0.002    0.000 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
        2    0.001    0.000    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       50    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      100    0.000    0.000    0.001    0.000 _py_warnings.py:294(simplefilter)
      300    0.000    0.000    0.001    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_7

```text
         21866 function calls in 0.079 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.079    0.079 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.002    0.002    0.079    0.079 surrogate.py:1721(__call__)
        1    0.000    0.000    0.072    0.072 surrogate.py:938(__call__)
        1    0.034    0.034    0.052    0.052 surrogate.py:741(_coorbital_to_inertial_frame)
        5    0.000    0.000    0.020    0.004 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.019    0.004 surrogate.py:416(__call__)
       10    0.000    0.000    0.019    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.018    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.018    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:96(__call__)
        6    0.000    0.000    0.016    0.003 surrogate.py:90(_splinterp_Cwrapper)
       50    0.000    0.000    0.016    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.016    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
        4    0.000    0.000    0.009    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.008    0.002    0.009    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
        2    0.007    0.004    0.007    0.004 spline_interp_Cwrapper.py:50(interpolate)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
        1    0.000    0.000    0.003    0.003 {method 'update' of 'dict' objects}
       11    0.003    0.000    0.003    0.000 surrogate.py:2126(<genexpr>)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
       14    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
        5    0.001    0.000    0.001    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
        3    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
```

##### NRHybSur2dq15 / mks_dt_0.0001220703125_flow_20

```text
         21862 function calls in 0.026 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.026    0.026 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.026    0.026 surrogate.py:1721(__call__)
        1    0.000    0.000    0.026    0.026 surrogate.py:938(__call__)
        5    0.000    0.000    0.018    0.004 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.018    0.004 surrogate.py:416(__call__)
       10    0.000    0.000    0.018    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
        1    0.003    0.003    0.007    0.007 surrogate.py:741(_coorbital_to_inertial_frame)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
        6    0.000    0.000    0.003    0.001 surrogate.py:90(_splinterp_Cwrapper)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
        4    0.000    0.000    0.002    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.002    0.000    0.002    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       50    0.001    0.000    0.001    0.000 kernels.py:1525(__call__)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
       50    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      100    0.000    0.000    0.001    0.000 _py_warnings.py:294(simplefilter)
     2950    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.01

```text
         21849 function calls in 0.026 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.026    0.026 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.026    0.026 surrogate.py:1721(__call__)
        1    0.000    0.000    0.026    0.026 surrogate.py:938(__call__)
        5    0.000    0.000    0.018    0.004 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.018    0.004 surrogate.py:416(__call__)
       10    0.000    0.000    0.018    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
        1    0.004    0.004    0.008    0.008 surrogate.py:741(_coorbital_to_inertial_frame)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
        6    0.000    0.000    0.004    0.001 surrogate.py:90(_splinterp_Cwrapper)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
        4    0.000    0.000    0.002    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.002    0.001    0.002    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
       50    0.001    0.000    0.001    0.000 kernels.py:1525(__call__)
        2    0.001    0.001    0.001    0.001 spline_interp_Cwrapper.py:50(interpolate)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
       50    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      100    0.000    0.000    0.001    0.000 _py_warnings.py:294(simplefilter)
      300    0.000    0.000    0.001    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur2dq15 / geom_dt_0.1_flow_0.002

```text
         21849 function calls in 0.360 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.360    0.360 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.020    0.020    0.360    0.360 surrogate.py:1721(__call__)
        1    0.000    0.000    0.327    0.327 surrogate.py:938(__call__)
        1    0.213    0.213    0.308    0.308 surrogate.py:741(_coorbital_to_inertial_frame)
        6    0.000    0.000    0.087    0.015 surrogate.py:90(_splinterp_Cwrapper)
        4    0.000    0.000    0.048    0.012 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.047    0.012    0.048    0.012 spline_interp_Cwrapper.py:123(interpolate_many_complex)
        2    0.039    0.020    0.039    0.020 spline_interp_Cwrapper.py:50(interpolate)
        5    0.000    0.000    0.019    0.004 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.019    0.004 surrogate.py:416(__call__)
       10    0.000    0.000    0.019    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.016    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.016    0.000 evaluate_fit.py:128(GPR_predict)
        5    0.012    0.002    0.012    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
        4    0.003    0.001    0.003    0.001 _function_base_impl.py:1402(diff)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
        2    0.002    0.001    0.002    0.001 surrogate.py:731(_search_omega)
       14    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
        2    0.001    0.001    0.001    0.001 _function_base_impl.py:5577(append)
        1    0.001    0.001    0.001    0.001 {built-in method numpy.arange}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.01

```text
         21849 function calls in 0.024 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.024    0.024 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.024    0.024 surrogate.py:1721(__call__)
        1    0.000    0.000    0.024    0.024 surrogate.py:938(__call__)
        5    0.000    0.000    0.018    0.004 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.018    0.004 surrogate.py:416(__call__)
       10    0.000    0.000    0.018    0.002 surrogate.py:291(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.015    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
        1    0.002    0.002    0.006    0.006 surrogate.py:741(_coorbital_to_inertial_frame)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
        6    0.000    0.000    0.003    0.000 surrogate.py:90(_splinterp_Cwrapper)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
        4    0.000    0.000    0.002    0.000 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.002    0.000    0.002    0.000 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
       14    0.001    0.000    0.001    0.000 {method 'dot' of 'numpy.ndarray' objects}
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
        2    0.001    0.000    0.001    0.000 spline_interp_Cwrapper.py:50(interpolate)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
       50    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
      100    0.000    0.000    0.001    0.000 fromnumeric.py:2304(sum)
      150    0.000    0.000    0.001    0.000 base.py:603(__sklearn_tags__)
      100    0.000    0.000    0.001    0.000 _py_warnings.py:294(simplefilter)
```

##### NRHybSur2dq15 / geom_dt_0.5_flow_0.002

```text
         21849 function calls in 0.084 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.085    0.085 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.003    0.003    0.085    0.085 surrogate.py:1721(__call__)
        1    0.000    0.000    0.080    0.080 surrogate.py:938(__call__)
        1    0.040    0.040    0.061    0.061 surrogate.py:741(_coorbital_to_inertial_frame)
        5    0.000    0.000    0.019    0.004 surrogate.py:424(_eval_sur)
        5    0.000    0.000    0.019    0.004 surrogate.py:416(__call__)
       10    0.000    0.000    0.019    0.002 surrogate.py:291(__call__)
        6    0.000    0.000    0.019    0.003 surrogate.py:90(_splinterp_Cwrapper)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:205(__call__)
       50    0.000    0.000    0.017    0.000 nodeFunction.py:140(__call__)
       50    0.000    0.000    0.016    0.000 nodeFunction.py:96(__call__)
       50    0.000    0.000    0.016    0.000 evaluate_fit.py:247(gprfitEvaluator)
       50    0.000    0.000    0.015    0.000 evaluate_fit.py:128(GPR_predict)
        4    0.000    0.000    0.010    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        4    0.010    0.002    0.010    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      100    0.000    0.000    0.010    0.000 validation.py:2793(validate_data)
       50    0.000    0.000    0.009    0.000 _gpr.py:373(predict)
        2    0.008    0.004    0.008    0.004 spline_interp_Cwrapper.py:50(interpolate)
      100    0.001    0.000    0.008    0.000 validation.py:725(check_array)
       50    0.000    0.000    0.006    0.000 _base.py:297(predict)
       50    0.000    0.000    0.006    0.000 _base.py:287(_decision_function)
       50    0.000    0.000    0.004    0.000 kernels.py:833(__call__)
      100    0.001    0.000    0.003    0.000 validation.py:103(_assert_all_finite)
       50    0.000    0.000    0.003    0.000 kernels.py:931(__call__)
      300    0.001    0.000    0.002    0.000 validation.py:371(_num_samples)
        5    0.002    0.000    0.002    0.000 {method 'conjugate' of 'numpy.ndarray' objects}
       14    0.002    0.000    0.002    0.000 {method 'dot' of 'numpy.ndarray' objects}
       50    0.001    0.000    0.002    0.000 kernels.py:1525(__call__)
     1952    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
        4    0.001    0.000    0.001    0.000 _function_base_impl.py:1402(diff)
      100    0.000    0.000    0.001    0.000 _array_api.py:857(_asarray_with_order)
       50    0.000    0.000    0.001    0.000 kernels.py:1239(__call__)
      150    0.000    0.000    0.001    0.000 _tags.py:250(get_tags)
      150    0.000    0.000    0.001    0.000 _py_warnings.py:320(_add_filter)
       50    0.000    0.000    0.001    0.000 validation.py:1621(check_is_fitted)
       50    0.000    0.000    0.001    0.000 kernels.py:1369(__call__)
      150    0.000    0.000    0.001    0.000 base.py:1166(__sklearn_tags__)
       50    0.000    0.000    0.001    0.000 _py_warnings.py:254(filterwarnings)
      200    0.000    0.000    0.001    0.000 _aliases.py:89(asarray)
        2    0.001    0.000    0.001    0.000 surrogate.py:731(_search_omega)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_7

```text
         81324 function calls in 0.185 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.186    0.186 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.002    0.002    0.186    0.186 surrogate.py:1721(__call__)
        1    0.000    0.000    0.180    0.180 surrogate.py:938(__call__)
       10    0.000    0.000    0.103    0.010 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.103    0.010 surrogate.py:416(__call__)
       17    0.000    0.000    0.102    0.006 surrogate.py:291(__call__)
      188    0.000    0.000    0.092    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.092    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.084    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.084    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.084    0.000 evaluate_fit.py:128(GPR_predict)
        1    0.054    0.054    0.077    0.077 surrogate.py:741(_coorbital_to_inertial_frame)
      376    0.002    0.000    0.054    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.049    0.000 _gpr.py:373(predict)
      376    0.006    0.000    0.044    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.033    0.000 _base.py:297(predict)
      188    0.001    0.000    0.033    0.000 _base.py:287(_decision_function)
       11    0.000    0.000    0.021    0.002 surrogate.py:90(_splinterp_Cwrapper)
      188    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
        9    0.000    0.000    0.015    0.002 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.015    0.002    0.015    0.002 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      188    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
        2    0.005    0.003    0.005    0.003 spline_interp_Cwrapper.py:50(interpolate)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
      376    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
```

##### NRHybSur3dq8_CCE / mks_dt_0.000244140625_flow_20

```text
         81320 function calls in 0.127 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.128    0.128 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.128    0.128 surrogate.py:1721(__call__)
        1    0.000    0.000    0.127    0.127 surrogate.py:938(__call__)
       10    0.000    0.000    0.108    0.011 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.108    0.011 surrogate.py:416(__call__)
       17    0.000    0.000    0.108    0.006 surrogate.py:291(__call__)
      188    0.000    0.000    0.095    0.001 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.095    0.001 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.087    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.087    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.087    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.054    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.051    0.000 _gpr.py:373(predict)
      376    0.006    0.000    0.044    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.033    0.000 _base.py:297(predict)
      188    0.002    0.000    0.033    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.022    0.000 kernels.py:833(__call__)
        1    0.011    0.011    0.019    0.019 surrogate.py:741(_coorbital_to_inertial_frame)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.016    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.012    0.001    0.012    0.001 {method 'dot' of 'numpy.ndarray' objects}
      188    0.004    0.000    0.010    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.007    0.000 {built-in method builtins.isinstance}
       11    0.000    0.000    0.006    0.001 surrogate.py:90(_splinterp_Cwrapper)
      188    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
        9    0.000    0.000    0.005    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.004    0.000    0.005    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
      376    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
     1504    0.002    0.000    0.004    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_7

```text
         81324 function calls in 0.237 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.238    0.238 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.003    0.003    0.238    0.238 surrogate.py:1721(__call__)
        1    0.000    0.000    0.225    0.225 surrogate.py:938(__call__)
        1    0.087    0.087    0.124    0.124 surrogate.py:741(_coorbital_to_inertial_frame)
       10    0.000    0.000    0.101    0.010 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.101    0.010 surrogate.py:416(__call__)
       17    0.000    0.000    0.100    0.006 surrogate.py:291(__call__)
      188    0.000    0.000    0.090    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.089    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.082    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.082    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.082    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.052    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.048    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.042    0.000 validation.py:725(check_array)
       11    0.000    0.000    0.034    0.003 surrogate.py:90(_splinterp_Cwrapper)
      188    0.000    0.000    0.032    0.000 _base.py:297(predict)
      188    0.001    0.000    0.032    0.000 _base.py:287(_decision_function)
        9    0.000    0.000    0.024    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.024    0.003    0.024    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.012    0.000 validation.py:371(_num_samples)
        2    0.010    0.005    0.010    0.005 spline_interp_Cwrapper.py:50(interpolate)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
        1    0.000    0.000    0.007    0.007 {method 'update' of 'dict' objects}
       18    0.007    0.000    0.007    0.000 surrogate.py:2126(<genexpr>)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.004    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8_CCE / mks_dt_0.0001220703125_flow_20

```text
         81320 function calls in 0.124 seconds

   Ordered by: cumulative time
   List reduced from 174 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.125    0.125 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.125    0.125 surrogate.py:1721(__call__)
        1    0.000    0.000    0.124    0.124 surrogate.py:938(__call__)
       10    0.000    0.000    0.102    0.010 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.102    0.010 surrogate.py:416(__call__)
       17    0.000    0.000    0.102    0.006 surrogate.py:291(__call__)
      188    0.000    0.000    0.092    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.091    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.084    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.084    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.083    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.053    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.049    0.000 _gpr.py:373(predict)
      376    0.006    0.000    0.043    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.033    0.000 _base.py:297(predict)
      188    0.001    0.000    0.033    0.000 _base.py:287(_decision_function)
        1    0.013    0.013    0.022    0.022 surrogate.py:741(_coorbital_to_inertial_frame)
      188    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
       11    0.000    0.000    0.007    0.001 surrogate.py:90(_splinterp_Cwrapper)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
        9    0.000    0.000    0.005    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.005    0.001    0.005    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
      376    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
     1504    0.002    0.000    0.004    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.01

```text
         81300 function calls in 0.127 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.128    0.128 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.128    0.128 surrogate.py:1721(__call__)
        1    0.000    0.000    0.127    0.127 surrogate.py:938(__call__)
       10    0.000    0.000    0.102    0.010 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.102    0.010 surrogate.py:416(__call__)
       17    0.000    0.000    0.102    0.006 surrogate.py:291(__call__)
      188    0.000    0.000    0.092    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.091    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.084    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.084    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.083    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.053    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.049    0.000 _gpr.py:373(predict)
      376    0.006    0.000    0.043    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.033    0.000 _base.py:297(predict)
      188    0.001    0.000    0.033    0.000 _base.py:287(_decision_function)
        1    0.015    0.015    0.025    0.025 surrogate.py:741(_coorbital_to_inertial_frame)
      188    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.015    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
       11    0.000    0.000    0.008    0.001 surrogate.py:90(_splinterp_Cwrapper)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
        9    0.000    0.000    0.006    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.005    0.001    0.006    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      188    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
      376    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
     1504    0.001    0.000    0.004    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8_CCE / geom_dt_0.1_flow_0.002

```text
         81300 function calls in 0.700 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.701    0.701 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.029    0.029    0.701    0.701 surrogate.py:1721(__call__)
        1    0.000    0.000    0.655    0.655 surrogate.py:938(__call__)
        1    0.386    0.386    0.554    0.554 surrogate.py:741(_coorbital_to_inertial_frame)
       11    0.000    0.000    0.157    0.014 surrogate.py:90(_splinterp_Cwrapper)
        9    0.000    0.000    0.102    0.011 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.101    0.011    0.102    0.011 spline_interp_Cwrapper.py:123(interpolate_many_complex)
       10    0.000    0.000    0.100    0.010 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.100    0.010 surrogate.py:416(__call__)
       17    0.000    0.000    0.100    0.006 surrogate.py:291(__call__)
      188    0.000    0.000    0.090    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.089    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.082    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.082    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.082    0.000 evaluate_fit.py:128(GPR_predict)
        2    0.055    0.027    0.055    0.028 spline_interp_Cwrapper.py:50(interpolate)
      376    0.002    0.000    0.052    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.048    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.042    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.032    0.000 _base.py:297(predict)
      188    0.001    0.000    0.032    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
        7    0.017    0.002    0.017    0.002 {method 'conjugate' of 'numpy.ndarray' objects}
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
        4    0.006    0.001    0.006    0.001 _function_base_impl.py:1402(diff)
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.01

```text
         81300 function calls in 0.120 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.121    0.121 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.000    0.000    0.121    0.121 surrogate.py:1721(__call__)
        1    0.000    0.000    0.120    0.120 surrogate.py:938(__call__)
       10    0.000    0.000    0.103    0.010 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.103    0.010 surrogate.py:416(__call__)
       17    0.000    0.000    0.102    0.006 surrogate.py:291(__call__)
      188    0.000    0.000    0.092    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.092    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.085    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.084    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.084    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.054    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.049    0.000 _gpr.py:373(predict)
      376    0.006    0.000    0.044    0.000 validation.py:725(check_array)
      188    0.000    0.000    0.033    0.000 _base.py:297(predict)
      188    0.001    0.000    0.033    0.000 _base.py:287(_decision_function)
      188    0.001    0.000    0.020    0.000 kernels.py:833(__call__)
        1    0.010    0.010    0.017    0.017 surrogate.py:741(_coorbital_to_inertial_frame)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.013    0.000 validation.py:371(_num_samples)
       21    0.009    0.000    0.009    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
       11    0.000    0.000    0.006    0.001 surrogate.py:90(_splinterp_Cwrapper)
      188    0.001    0.000    0.006    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.006    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
        9    0.000    0.000    0.005    0.001 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.004    0.000    0.005    0.001 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.005    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      376    0.001    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
     1504    0.001    0.000    0.004    0.000 _config.py:35(get_config)
```

##### NRHybSur3dq8_CCE / geom_dt_0.5_flow_0.002

```text
         81300 function calls in 0.254 seconds

   Ordered by: cumulative time
   List reduced from 172 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.254    0.254 benchmark_surrogate_evaluations.py:269(evaluate_case)
        1    0.007    0.007    0.254    0.254 surrogate.py:1721(__call__)
        1    0.000    0.000    0.244    0.244 surrogate.py:938(__call__)
        1    0.102    0.102    0.143    0.143 surrogate.py:741(_coorbital_to_inertial_frame)
       10    0.000    0.000    0.100    0.010 surrogate.py:424(_eval_sur)
       10    0.000    0.000    0.100    0.010 surrogate.py:416(__call__)
       17    0.000    0.000    0.100    0.006 surrogate.py:291(__call__)
      188    0.000    0.000    0.090    0.000 nodeFunction.py:205(__call__)
      188    0.002    0.000    0.089    0.000 nodeFunction.py:110(__call__)
      188    0.000    0.000    0.082    0.000 nodeFunction.py:96(__call__)
      188    0.000    0.000    0.082    0.000 evaluate_fit.py:247(gprfitEvaluator)
      188    0.001    0.000    0.081    0.000 evaluate_fit.py:128(GPR_predict)
      376    0.002    0.000    0.052    0.000 validation.py:2793(validate_data)
      188    0.002    0.000    0.048    0.000 _gpr.py:373(predict)
      376    0.005    0.000    0.042    0.000 validation.py:725(check_array)
       11    0.000    0.000    0.038    0.003 surrogate.py:90(_splinterp_Cwrapper)
      188    0.000    0.000    0.032    0.000 _base.py:297(predict)
      188    0.001    0.000    0.032    0.000 _base.py:287(_decision_function)
        9    0.000    0.000    0.026    0.003 surrogate.py:85(_splinterp_Cwrapper_many_complex)
        9    0.026    0.003    0.026    0.003 spline_interp_Cwrapper.py:123(interpolate_many_complex)
      188    0.001    0.000    0.019    0.000 kernels.py:833(__call__)
      376    0.004    0.000    0.016    0.000 validation.py:103(_assert_all_finite)
      188    0.001    0.000    0.014    0.000 kernels.py:931(__call__)
     1128    0.005    0.000    0.012    0.000 validation.py:371(_num_samples)
        2    0.011    0.006    0.011    0.006 spline_interp_Cwrapper.py:50(interpolate)
       21    0.010    0.000    0.010    0.000 {method 'dot' of 'numpy.ndarray' objects}
      188    0.003    0.000    0.008    0.000 kernels.py:1525(__call__)
     7334    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      188    0.001    0.000    0.005    0.000 kernels.py:1239(__call__)
      564    0.000    0.000    0.005    0.000 _tags.py:250(get_tags)
      376    0.001    0.000    0.005    0.000 _array_api.py:857(_asarray_with_order)
      564    0.002    0.000    0.005    0.000 _py_warnings.py:320(_add_filter)
      188    0.000    0.000    0.005    0.000 kernels.py:1369(__call__)
      188    0.000    0.000    0.004    0.000 validation.py:1621(check_is_fitted)
      564    0.001    0.000    0.004    0.000 base.py:1166(__sklearn_tags__)
      752    0.001    0.000    0.004    0.000 _aliases.py:89(asarray)
    11092    0.004    0.000    0.004    0.000 {built-in method builtins.hasattr}
      376    0.000    0.000    0.004    0.000 _py_warnings.py:294(simplefilter)
     1128    0.001    0.000    0.004    0.000 _array_api.py:331(get_namespace)
      564    0.001    0.000    0.004    0.000 base.py:603(__sklearn_tags__)
```
