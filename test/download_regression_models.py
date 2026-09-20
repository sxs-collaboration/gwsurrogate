"""Ensure verified local copies of all models tested in test_model_regression.py.

Existing files with matching catalog checksums are reused. This is useful for
continuous integration, including when model files are restored from a cache."""

import gwsurrogate as gws

models = ['SpEC_q1_10_NoSpin_linear_alt',
'NRHybSur3dq8',
'NRHybSur3dq8_CCE',
'SpEC_q1_10_NoSpin',
'SpEC_q1_10_NoSpin_linear',
'NRSur7dq4',
'NRSur7dq4v2',
'NRHybSur2dq15',
'SEOBNRv4PHMSur',
'EOBNRv2',
'EMRISur1dq1e4',
'BHPTNRSur1dq1e4'
]

if __name__ == '__main__':
  for model in models:
    gws.catalog.pull(model)
