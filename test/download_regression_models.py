""" download all models to be tested in test_model_regression.py

This is useful to do when using continuous integration. """


import gwsurrogate as gws

models = ['SpEC_q1_10_NoSpin_linear_alt',
'NRHybSur3dq8',
'SpEC_q1_10_NoSpin',
'SpEC_q1_10_NoSpin_linear'
]

for model in models:
  gws.catalog.pull(model)
