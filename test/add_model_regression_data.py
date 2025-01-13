"""
When adding a new model, its good to also setup a model regression test.

Here are the steps for doing this (the last step is running this script)


First make sure the current tests run on your laptop
====================================================
Download all model data

>>>  python download_regression_models.py  # run from tests folder

The run all of the tests

>>> pytest -v -s 

This will also download a file called model_regression_data.h5,
which will be used later on.


Setup tests for the new model (after you have implemented the model)
====================================================================
1) Add your model to download_regression_models.py
2) Add model to interface list in test_model_regression.py
3) Write a routine to decide at which points to sample the model in test_model_regression.py


Generate new model regression data
===================================
1) Generate the new regression data by running

>>> python test_model_regression.py  #run from tests folder

which will produce a file called model_regression_data_new.h5. This file will contain
regression data for both your new model as well as previous models. However, we
don't want to overwrite previous model regression data. 

2) The last step is to copy your model-specific regression data into
the existing file model_regression_data.h5. This way we don't overwrite the previous 
regression data.

>>> python add_model_regression_data.py

After running this step (Note: make sure to change the value
of NEW_MODEL in this script), model_regression_data.h5 
should have the new model's regression data. 

3) model_regression_data_new.h5 can be deleted
4) run check_model_regression_data.py to confirm that your data has been 
correctly added and the old data is unchanged.


History of model regression data files
======================================
model_regression_data.h5 files, newest first...

* Added SEOBNRv4PHMSur model (current)
  md5sum: e7f4cbca43b8a8a6e914153cfbcc9aea
* Added NRHybSur3dq8_CCE model (used until around 1/13/2024)
  md5sum: 4616a1e661e643f51acc857cb0534156
"""

import h5py
import numpy as np
import shutil

original_regression_data = "model_regression_data.h5"
extra_regression_data    = "model_regression_data_new.h5"
new_model = "SEOBNRv4PHMSur" # name of new model with regression data in fp_new but not fp_old

shutil.copyfile(original_regression_data, "model_regression_data-original.h5")


fp_new_data = h5py.File(extra_regression_data,'r')
fp_old_data = h5py.File(original_regression_data,'r+')


for model in fp_new_data.keys():
    if model in fp_old_data.keys():
        print("model %s already exists in regression dataset. Skipping...."%model)
    else:
        print("model %s does not exist in regression dataset. Adding...."%model)
        fp_new_data.copy(model,fp_old_data)


fp_new_data.close()
fp_old_data.close()


# the file model_regression_data should have the new model regression data for new_model
# we now check that the copying went OK.
# 
# Check 1: the new_model is in the new datafile but not the old one
# Check 2: the original data is unchanged

## --- test that the copying worked correctly --- ##


fp_new = h5py.File("model_regression_data.h5")
fp_old = h5py.File("model_regression_data-original.h5",'r')


print("models in new datafile")
for model in fp_new.keys(): # loop over models
    print(model)
print("---------------")
print("models in old datafile")
for model in fp_old.keys(): # loop over models
    print(model)

for model in fp_new.keys(): # loop over models
    if model != new_model:
        model_regession_data = fp_new[model]
        for paramter in model_regession_data.keys(): # loop over parameters
            model_regession_data_i = model_regession_data[paramter]
            for data in model_regession_data_i.keys(): # loop over data for each parameter
                if data != "parameter" and data != "time":
                    x = '/'+model+'/'+paramter+'/'+data
                    y1 = model_regession_data_i[data][:,:]
                    y2 = fp_old[x][:,:]
                    y = y1 - y2
                    if np.max(np.abs(y)) == 0.0:
                        pass
                    else:
                        print("New and old regression data doesn't agree.")
                        print(x)
                        print(y)
    else:
        print("model %s doesn't exist in original dataset"%new_model)
