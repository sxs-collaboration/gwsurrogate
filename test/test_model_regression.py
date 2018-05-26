"""
Test every gwsurrogate model.

Each model should test the following at a small handful of parameter values:

* all modes 
* summation of all modes
* each of the two above, for dimensionless and physics waveforms

Before running this script (as a test), generate regression data

>>> python test_model_regression.py

from the folder test
"""

# TODO: new and old surroagte interfaces should be similar enough to avoid
#       model-specific cases like below


from __future__ import division
import numpy as np
import gwsurrogate as gws
from gwsurrogate.new import surrogate 
import h5py
import subprocess

# set global tolerances for floating point comparisons (see np.testing.assert_allclose)
#atol = 0.0
#rtol = 1.e-11


def test_model_regression(generate_regression_data=False):
  """ If generate_regression_data = True, this script will generate 
  an hdf5 file to diff against. 

  If generate_regression_data = False, this script will compare 
  model evaluations to the hdf5 file produced when True. """

  if generate_regression_data:
    h5_file = "regression_data.h5"
    print("Generating regression data file... Make sure this step is done BEFORE making any code changes!\n")
  else:
    h5_file = "test/comparison_data.h5" # assumes py.test runs from project-level folder

  # remove models if you don't have them
  dont_test = ["NRSur4d2s_TDROM_grid12",
               "NRSur4d2s_FDROM_grid12",
               #"SpEC_q1_10_NoSpin_linear_alt",
               #"SpEC_q1_10_NoSpin_linear",
               "EOBNRv2", #TODO: this is two surrogates in one. Break up?
               #"SpEC_q1_10_NoSpin",
               #"EOBNRv2_tutorial"
               ]

  # Common directory where all surrogates are assumed to be located
  surrogate_path = gws.catalog.download_path()
  print("data assumed to be in the path %s. If your surrogate are somewehre else, change the path"%surrogate_path)

  # repeatability needed for regression tests to make sense 
  np.random.seed(0)

  # for each model, associate its surrogate data file
  models = [model for model in gws.catalog._surrogate_world]
  models_to_test = {}
  for model in models:
    surrogate_data = surrogate_path+gws.catalog._surrogate_world[model][0].split("/")[-1]
    models_to_test[model] = surrogate_data
 
  # also test the tutorial surrogate
  models_to_test["EOBNRv2_tutorial"] = gws.__path__[0] + "/../tutorial/TutorialSurrogate/EOB_q1_2_NoSpin_Mode22/"

  # remove models from testing...
  for i in dont_test:
    models_to_test.pop(i)

  # Old surrogate interface
  surrogate_old_interface = ["SpEC_q1_10_NoSpin","EOBNRv2_tutorial","EOBNRv2","SpEC_q1_10_NoSpin_linear"]

  fp = h5py.File(h5_file,"w")

  # for each model, select three random points to evalaute at
  for model, datafile in models_to_test.items():

    print("Generating regression data for model = %s"%model)
    print(datafile)

    if model in surrogate_old_interface:
      sur = gws.EvaluateSurrogate(datafile)
    else:
      sur = surrogate.FastTensorSplineSurrogate()
      sur.load(datafile)
        
    p_mins = sur.param_space.min_vals()
    p_maxs = sur.param_space.max_vals()
    print("parameter minimum values",p_mins)
    print("parameter maximum values",p_maxs)
    param_samples = np.random.uniform(p_mins[0], p_maxs[0],size=3)
    model_grp = fp.create_group(model)
    for i, ps in enumerate(param_samples):
      if model in surrogate_old_interface:
        modes, t, hp, hc = sur(q=ps,mode_sum=False,fake_neg_modes=True)
      else:
        h= sur([ps])
        h_np = [h[mode] for mode in sur.mode_list]
        h_np = np.vstack(h_np)
        hp = np.real(h_np)
        hc = np.imag(h_np)
      samplei = model_grp.create_group("parameter"+str(i))
      samplei.create_dataset("parameter",data=ps)
      samplei.create_dataset("hp",data=hp)
      samplei.create_dataset("hc",data=hc)
  fp.close()
  
  if not generate_regression_data:
    process = subprocess.Popen(["h5diff", "test/regression_data.h5",h5_file],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    returncode = process.wait()
    stdout, stderr = process.communicate()
    if returncode == 0:
        assert(True)
    else:
        print(stdout)
        print(stderr)
        assert(False)
    


#------------------------------------------------------------------------------
if __name__ == "__main__":
  test_model_regression(generate_regression_data=True)
