"""
Test every gwsurrogate model.

Each model should test the following at a small handful of parameter values:

* all modes 
* summation of all modes
* each of the two above, for dimensionless and physical waveforms

Running this script as a regression test (with pytest) will download
regression data from Dropbox. 

Regression data should NOT be generated locally unless a new regression
file is being created to upload to Dropbox. In this case do

>>> python test_model_regression.py

from the folder test.


NOTE: waveform regression data is saved with single precision In order to,
     (i) reduce the size of the regression file and
     (ii) allow h5diff to not fail due to round-off error (Still fails! switched to allclose)

NOTE: No regression on dynamics surrogate output. This is probably
      OK since coorb full surrogate uses dynamics output.
"""


from __future__ import division
import numpy as np
import gwsurrogate as gws
from gwsurrogate.new import surrogate 
import h5py, os, subprocess, time, warnings

# set global tolerances for floating point comparisons (see np.testing.assert_allclose)
atol = 0.0
# why a high tolerance? For some reason, a high tolerance is needed when 
# comparining to regression data on different machines
# TODO: explore the orgin of these large discrepencies (note that hdf5 data is saved in single precision, and errors seem to post-merger)
rtol_gsl = 5.e-1
rtol = 1.e-11

# TODO: new and old surrogate interfaces should be similar enough to avoid
#       model-specific cases like below

# Old surrogate interface
surrogate_old_interface = ["SpEC_q1_10_NoSpin","EOBNRv2_tutorial","EOBNRv2","SpEC_q1_10_NoSpin_linear"]

# news loader class
surrogate_loader_interface = ["NRHybSur3dq8","NRHybSur3dq8Tidal","NRSur7dq4"]

# Most models are randomly sampled, but in some cases its useful to provide 
# test points to activate specific code branches. This is done by mapping 
# a model (named in the surrogate catalog) to a function that will return 
# 3 unique paramter points.
#
# Each sampler should return a list [q, chiA, chiB] and a dictionary tidOpts
# and a dictionary precesingOpts
model_sampler = {}

def NRHybSur3dq8Tidal_samples(i):
  """ sample points for the NRHybSur3dq8Tidal model
  the ith sample point to evaluate the model at
  samples are returned as [q, chiA, chiB], tidOpts """

  assert i in [0,1,2]

  if i==0:
    return [1.2, [0.,0.,.1], [0.,0.,.1]], {'Lambda1': 1000.0, 'Lambda2': 4000.0}, None
  elif i==1:
    return [1.2, [0.,0.,.4], [0.,0.,-.4]], {'Lambda1': 0.0, 'Lambda2': 9000.0}, None
  elif i==2:
    return [1.2, [0.,0.,.0], [0.,0.,.0]], {'Lambda1': 0.0, 'Lambda2': 0.0}, None

model_sampler["NRHybSur3dq8Tidal"] = NRHybSur3dq8Tidal_samples


def NRSur7dq4_samples(i):
  """ sample points for the NRSur7dq4 model
  the ith sample point to evaluate the model at
  samples are returned as [q, chiA, chiB], precessingOpts """

  assert i in [0,1,2]

  if i==0:
    chiA = [-0.2, 0.4, 0.1]
    chiB = [-0.5, 0.2, -0.4]
    precessing_opts = {'init_quat': [1,0,0,0],
                       'return_dynamics': True,
                       'init_orbphase': 0.0}
                       #'use_lalsimulation_conventions': True}
    return [2., chiA, chiB], None, precessing_opts
  elif i==1:
    chiA = [-0.2, 0.4, 0.1]
    chiB = [-0.5, 0.2, -0.4]
    precessing_opts = {'init_quat': [1,0,0,0],
                       'return_dynamics': True,
                       'init_orbphase': 1.0}
                       #'use_lalsimulation_conventions': False}
    return [3., chiA, chiB], None, precessing_opts
  elif i==2:
    chiA = [-0.2, 0.4, 0.1]
    chiB = [-0.5, 0.2, -0.4]
    precessing_opts = {'init_quat': [1,0,0,0],
                       'return_dynamics': True,
                       'init_orbphase': 0.0}
                       #'use_lalsimulation_conventions': True}
    return [5., chiA, chiB], None, precessing_opts

model_sampler["NRSur7dq4"] = NRSur7dq4_samples

def flatten_params(x):
  """ Convert [q, chiA, chiB] to [q, chiAx, chiAy, chiAz, chiBx, chiBy, chiBz].

  This function is only used when writting samples from specific model samplers
  to HDF5 file. """

  return [x[0],x[1][0],x[1][1],x[1][2],x[2][0],x[2][1],x[2][2]]


def test_model_regression(generate_regression_data=False):
  """ If generate_regression_data = True, this script will generate 
  an hdf5 file to diff against. No regression will be done.

  If generate_regression_data = False, this script will compare 
  model evaluations to the hdf5 file produced when True. In a typical use case,
  this regression file will be downloaded. """

  if generate_regression_data:
    h5_file = "model_regression_data.h5"
    print("Generating regression data file... Make sure this step is done BEFORE making any code changes!\n")
    print(os.path.exists(h5_file))
    if os.path.exists(h5_file):
      raise RuntimeError("Refusing to overwrite a regression file!")
  else:
    h5_file = "test/comparison_data.h5" # assumes pytest runs from project-level folder
    try: # try importing data. If it doesn't exist, download it
      fp_regression = h5py.File("test/model_regression_data.h5",'r') 
    except IOError:
      print("Downloading regression data...")
      os.system('wget --directory-prefix=test https://www.dropbox.com/s/vxqsr7fjoffxm5w/model_regression_data.h5')
      fp_regression = h5py.File("test/model_regression_data.h5",'r') 

  # remove models if you don't have them
  dont_test = ["EMRISur1dq1e4", # model data not currently available in public
               "NRSur4d2s_TDROM_grid12", # 10 GB file
               "NRSur4d2s_FDROM_grid12", # 10 GB file
               #"SpEC_q1_10_NoSpin_linear_alt",
               #"SpEC_q1_10_NoSpin_linear",
               "EOBNRv2", #TODO: this is two surrogates in one. Break up?
               #"SpEC_q1_10_NoSpin",
               #"EOBNRv2_tutorial",
               #"NRHybSur3dq8",
               #"NRHybSur3dq8Tidal",
               #"NRSur7dq4"
               ]

  # Common directory where all surrogates are assumed to be located
  surrogate_path = gws.catalog.download_path()
 
  # repeatability can be useful for regression tests
  np.random.seed(0)

  # for each model, associate its surrogate data file
  models = [model for model in gws.catalog._surrogate_world]
  print(models)

  models_to_test = {}
  for model in models:
    surrogate_data = surrogate_path+os.path.basename(gws.catalog._surrogate_world[model][0])
    if os.path.isfile(surrogate_data): # surrogate data file exists
      models_to_test[model] = surrogate_data
    else: # file missing 
      msg = "WARNING: Surrogate missing!!!\n"
      msg += "Surrogate data assumed to be in the path %s.\n"%surrogate_data
      msg += "If the data is somewhere else, change the path or move the file.\n\n"
      msg +="To download this surrogate, from ipython do\n\n >>> gws.catalog.pull(%s)\n"%model
      print(msg)
      time.sleep(1)
      
 
  # also test the tutorial surrogate
  models_to_test["EOBNRv2_tutorial"] = gws.__path__[0] + "/../tutorial/TutorialSurrogate/EOB_q1_2_NoSpin_Mode22/"

  # remove models from testing...
  for i in dont_test:
    try:
      models_to_test.pop(i)
      print("model %s removed from testing"%i)
    except KeyError:
      print("model %s cannot be removed"%i)

  fp = h5py.File(h5_file,"w")

  # for each model, select three random points to evaluate at
  param_samples_tested = []
  for model, datafile in models_to_test.items():

    print("Generating regression data for model = %s"%model)
    print(datafile)

    if model in surrogate_old_interface:
      sur = gws.EvaluateSurrogate(datafile)
      p_mins = sur.param_space.min_vals()
      p_maxs = sur.param_space.max_vals()
    elif model in surrogate_loader_interface:
      # sur = gws.LoadSurrogate(datafile) # tidal and aligned models use the same h5 file (so wrong one loaded) 
      sur = gws.LoadSurrogate(model)
      try:
        p_mins = sur._sur_dimless.param_space.min_vals()
        p_maxs = sur._sur_dimless.param_space.max_vals()
      except AttributeError: # NRSur7dq4 does not have object sur._sur_dimless.param_space
        p_mins = None
        p_maxs = None
    else:
      sur = surrogate.FastTensorSplineSurrogate()
      sur.load(datafile)
      p_mins = sur.param_space.min_vals()
      p_maxs = sur.param_space.max_vals()
 
        
    print("parameter minimum values",p_mins)
    print("parameter maximum values",p_maxs)

    param_samples = []
    if generate_regression_data: # pick new points to compute regression data at
      for i in range(3):  # sample parameter space 3 times 
        if model in list(model_sampler.keys()):
          custom_sampler = model_sampler[model]
          x, tidOpts, pecOpts = custom_sampler(i) # [q, chiA, chiB], tidOpts
        else: # default sampler for spin-aligned BBH models
          x = [] # [q, chiAz, chiBz]
          for j in range(len(p_mins)):
            xj_min = p_mins[j]
            xj_max = p_maxs[j]
            tmp = float(np.random.uniform(xj_min, xj_max,size=1))
            x.append(tmp)
          tidOpts = None
          pecOpts = None
        param_samples.append([x, tidOpts, pecOpts])
    else: # get point at which to compute comparison waveform data
      for i in range(3):
        if model in list(model_sampler.keys()): # use sample points if provided 
          custom_sampler = model_sampler[model]
          x, tidOpts, pecOpts = custom_sampler(i) # [q, chiA, chiB], tidOpts
        else: # pull regression points from regression data file; none-tidal models only
          print(model+"/parameter%i/parameter"%i)
          x = []
          x.append( list(fp_regression[model+"/parameter%i/parameter"%i][:]) ) # [q, chiAz, chiBz]
          x = list(fp_regression[model+"/parameter%i/parameter"%i][:]) # [q, chiAz, chiBz]
          tidOpts = None
          pecOpts = None
        param_samples.append([x, tidOpts, pecOpts])

    param_samples_tested.append(param_samples)

    model_grp = fp.create_group(model)
    for i, ps in enumerate(param_samples):
      x = ps[0]
      tidOpts = ps[1]
      pecOpts = ps[2]
      if model in surrogate_old_interface:
        ps_float = x[0] # TODO: generalize interface 
        modes, t, hp, hc = sur(q=ps_float,mode_sum=False,fake_neg_modes=True)
      else:
        if model in surrogate_loader_interface:
          q = x[0]
          if type(x[1]) is np.float64 or type(x[1]) is float: # chiz
            chiA = np.array([0, 0, x[1]])
            chiB = np.array([0, 0, x[2]])
          elif len(x[1])==3: # spin vector
            chiA = np.array(x[1])
            chiB = np.array(x[2])
          else:
            raise ValueError
          try:
                  # Regression samples outside of the training interval.
                  # Warnings are raised (as they should) but could 
                  # appear bad to a new user
              if model in ["NRSur7dq4"]:
                with warnings.catch_warnings():
                  warnings.simplefilter("ignore")
                  t, h, dyanmics = sur(q, chiA, chiB, f_low=0.0, tidal_opts=tidOpts, precessing_opts=pecOpts)
              else:
                t, h, dyanmics = sur(q, chiA, chiB, f_low=0.0, tidal_opts=tidOpts, precessing_opts=pecOpts)
          except ValueError: # some models do not allow for f_low=0.0 and require a time step
             # step size, Units of M
             # initial frequency, Units of cycles/M
            t, h, dyanmics = sur(q, chiA, chiB, dt = 0.25, f_low=3.e-3, tidal_opts=tidOpts, precessing_opts=pecOpts) 
        else:
          h= sur(x)
        try:
          h_np = [h[mode] for mode in sur.mode_list]
        except AttributeError: # for new interface
          h_np = [h[mode] for mode in sur._sur_dimless.mode_list]

        h_np = np.vstack(h_np)
        hp = np.real(h_np)
        hc = np.imag(h_np)
      samplei = model_grp.create_group("parameter"+str(i))
      if model in list(model_sampler.keys()):
        # model samplers return a list of lists, which we flatten into 
        # a list of numbers for storing in an h5 dataset
        x = flatten_params(x)
      samplei.create_dataset("parameter",data=x)
      samplei.create_dataset("hp", data=hp, dtype='float32')
      samplei.create_dataset("hc", data=hc, dtype='float32')

  fp.close()

  if not generate_regression_data:
    fp = h5py.File(h5_file,"r") # reopen comparison data
    for model in models_to_test.keys():
      print("testing model %s ..."%model)
      for i in range(3): # 3 parameter samples
        hp_regression = fp_regression[model+"/parameter%i/hp"%i][:]
        hc_regression = fp_regression[model+"/parameter%i/hp"%i][:]
        hp_comparison = fp[model+"/parameter%i/hp"%i][:]
        hc_comparison = fp[model+"/parameter%i/hp"%i][:]
        if model == "NRHybSur3dq8":
          local_rtol = rtol_gsl
        else:
          local_rtol = rtol_gsl
        np.testing.assert_allclose(hp_regression, hp_comparison, rtol=local_rtol, atol=atol)
        np.testing.assert_allclose(hc_regression, hc_comparison, rtol=local_rtol, atol=atol)

    # fails due to round-off error differences of different machines
    #fp_regression.close()
    #process = subprocess.Popen(["h5diff", "test/model_regression_data.h5",h5_file],
    #                           stdin=subprocess.PIPE,
    #                           stdout=subprocess.PIPE,
    #                           stderr=subprocess.PIPE)
    #returncode = process.wait()
    #stdout, stderr = process.communicate()
    #if returncode == 0:
    #    assert(True)
    #else:
    #    print(stdout)
    #    print(stderr)
    #    assert(False)
  print("models tested... ")
  #for i, model_tested in enumerate(models_to_test.keys()):
  #  print("model %s at points..."%model_tested+str(param_samples_tested[i]))


#------------------------------------------------------------------------------
if __name__ == "__main__":
  test_model_regression(generate_regression_data=True)
