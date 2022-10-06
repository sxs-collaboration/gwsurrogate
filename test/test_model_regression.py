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

NOTES
=====
(1) Single precision regression data

Waveform regression data is saved with single precision in order to,

     (i) reduce the size of the regression file and
     (ii) allow h5diff to not fail due to round-off error (Still fails! switched comparison to np.allclose)

this is OK because the models themselves are not accurate to better than single precision.


(2) Dynamics surrogate not (directly) tested

No regression on dynamics surrogate output. This is probably OK since coorb
full surrogate uses dynamics output.
"""


from __future__ import division
import numpy as np
import gwsurrogate as gws
from gwsurrogate.new import surrogate 
import h5py, os, subprocess, time, warnings


import hashlib

def md5(fname):
  """ Compute has from file. code taken from 
  https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file"""
  hash_md5 = hashlib.md5()
  with open(fname, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_md5.update(chunk)
  return hash_md5.hexdigest()

# set global tolerances for floating point comparisons. 
# From documentation on np.testing.assert_allclose
# the comparison is
#
#     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
#
# Setting atol=0 means we are only testing relative errors
#
atol = 0.0
# why a high tolerance? For some reason, a high tolerance is needed when 
# comparing to regression data on different machines
# TODO: explore the origin of these large discrepancies 
#       note that regression data model_regression_data.h5 is saved in single precision (see above)
#       largest relative errors seem to be post-merger
#       only seems to affect models that use gpr fits and/or gsl calls
rtol                   = 1.e-11
rtol_NRHybSur3dq8      = 2.e-5  # used for GPR-fit models "NRHybSur3dq8", "NRHybSur2dq15"
rtol_NRHybSur3dq8Tidal = 3.e-4
rtol_SpEC_q1_10_NoSpin_linear_alt = 3.e-8 # needed for (8,7) mode to pass. Other modes pass with 1e-11 tolerance

# TODO: new and old surrogate interfaces should be similar enough to avoid
#       model-specific cases like below

# Old surrogate interface
surrogate_old_interface = ["SpEC_q1_10_NoSpin","EOBNRv2_tutorial","EOBNRv2","SpEC_q1_10_NoSpin_linear","EMRISur1dq1e4","BHPTNRSur1dq1e4"]

# news loader class
surrogate_loader_interface = ["NRHybSur3dq8","NRHybSur3dq8Tidal","NRSur7dq4","NRHybSur2dq15"]

# Most models are randomly sampled, but in some cases its useful to provide 
# test points to activate specific code branches. This is done by mapping 
# a model (named in the surrogate catalog) to a function that will return 
# 3 unique parameter points.
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

def BHPTNRSur1dq1e4_samples(i):
  """ sample points for the BHPTNRSur1dq1e4 model """

  assert i in [0,1,2]

  if i==0:
    return [3.0, [0,0,0],[0,0,0]], None, None
  elif i==1:
    return [100.0, [0,0,0],[0,0,0]], None, None
  elif i==2:
    return [10000.0, [0,0,0],[0,0,0]], None, None

model_sampler["BHPTNRSur1dq1e4"] = BHPTNRSur1dq1e4_samples


def NRHybSur2dq15_samples(i):
  """ sample points for the NRHybSur2dq15 model """

  assert i in [0,1,2]

  if i==0:
    return [2.0, [0,0,-.4],[0,0,0]], None, None
  elif i==1:
    return [11.0, [0,0,.7],[0,0,0]], None, None
  elif i==2:
    return [18.0, [0,0,.4],[0,0,0]], None, None

model_sampler["NRHybSur2dq15"] = NRHybSur2dq15_samples


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
      # Old file (10/2022): https://www.dropbox.com/s/vxqsr7fjoffxm5w/model_regression_data.h5
      os.system('wget --directory-prefix=test https://www.dropbox.com/s/4zcse4ja5aw3n6s/model_regression_data.h5')
      fp_regression = h5py.File("test/model_regression_data.h5",'r') 
    regression_hash = md5("test/model_regression_data.h5")
    print("hash of model_regression_data.h5 is ",regression_hash)

  # remove models if you don't have them
  dont_test = [#"NRHybSur2dq15",
               #"BHPTNRSur1dq1e4",
               #"EMRISur1dq1e4",
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
      # Comment out following assert if experimenting with model-specific tests
      #assert(False)
      
 
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

    print("Generating data for model = %s"%model)
    print(datafile)

    if model in surrogate_old_interface:
      sur = gws.EvaluateSurrogate(datafile)
      # this is the surrogate's parameterization (e.g. log(q), \eta) region!
      # Technically should be using q. For SpEC_q1_10_NoSpin, Scott manually 
      # mapped eta to q, directly modifing the regression data file (10/5/2022)
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
 
        
    print("parameter minimum values for model %s"%model,p_mins)
    print("parameter maximum values for model %s"%model,p_maxs)

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
          modes = sur.mode_list
          h_np = [h[mode] for mode in modes]
        except AttributeError: # for new interface
          modes = sur._sur_dimless.mode_list
          h_np = [h[mode] for mode in modes]

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
      # added on 7/24/2022 -- not part of model_regression_data.h5
      # but is used in code logic below
      samplei.create_dataset("time", data=t, dtype='float32')
      samplei.create_dataset("modes", data=np.array(modes), dtype='int')

  fp.close()

  if not generate_regression_data:
    fp = h5py.File(h5_file,"r") # reopen comparison data
    for model in models_to_test.keys():
      print("testing model %s ..."%model)
      for i in range(3): # 3 parameter samples
        hp_regression = fp_regression[model+"/parameter%i/hp"%i][:]
        hc_regression = fp_regression[model+"/parameter%i/hc"%i][:]
        hp_comparison = fp[model+"/parameter%i/hp"%i][:]
        hc_comparison = fp[model+"/parameter%i/hc"%i][:]

        # only test data up to 90M past merger
        # Currently not used (but keep in case this is reactivated)
        # t_indx = np.argmin(np.abs(fp[model+"/parameter%i/time"%i][:] - 90) )
        t_indx = fp[model+"/parameter%i/time"%i][:].shape[0]  # use all time points

        # model-specific relative tolerances. This is needed because certain models
        # have dependencies (e.g. GSL or sklearn) that will break our tests!
        if model in ["NRHybSur3dq8", "NRHybSur2dq15"]:
          local_rtol = rtol_NRHybSur3dq8
        elif model == "NRHybSur3dq8Tidal":
          local_rtol = rtol_NRHybSur3dq8Tidal
        elif model == "SpEC_q1_10_NoSpin_linear_alt":
          local_rtol = rtol_SpEC_q1_10_NoSpin_linear_alt 
        else:
          local_rtol = rtol

        print("Model %s uses a relative error tolerance of %e"%(model,local_rtol))

        for j, mode in enumerate(fp[model+"/parameter%i/modes"%i][:]): # test mode-by-mode
          err_msg="Failed: model %s for mode index %i (ell = %i,m = %i)"%(model,j,mode[0],mode[1])

          # test hp and hc separately 
          # Note: because hp and hc oscillate about 0, this test can easily fail (relative error check) 
          # if hp/hc change by very small amounts
          #np.testing.assert_allclose(hp_regression[j,:t_indx], hp_comparison[j,:t_indx], rtol=local_rtol, atol=atol, err_msg=err_msg)
          #np.testing.assert_allclose(hc_regression[j,:t_indx], hc_comparison[j,:t_indx], rtol=local_rtol, atol=atol, err_msg=err_msg)

          # test complexified h.
          # This is a more robust relative-error test because the |h| does not pass through 0
          h_regression = hp_regression[j,:t_indx] + 1.0j*hc_regression[j,:t_indx]
          h_comparison = hp_comparison[j,:t_indx] + 1.0j*hc_comparison[j,:t_indx]
          np.testing.assert_allclose(h_regression, h_comparison, rtol=local_rtol, atol=atol, err_msg=err_msg)


        # when debugging the tests, it can be useful to dump the data
        #np.save("hp_regression-%i.npy"%i,hp_regression)
        #np.save("hp_comparison-%i.npy"%i,hp_comparison)
        #np.save("hc_regression-%i.npy"%i,hc_regression)
        #np.save("hc_comparison-%i.npy"%i,hc_comparison)


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
