"""
After adding a new model's regression data (see add_model_regression_data.py) 

run this script to check that the common hdf5 groups in the new and old hdf5
files are the same. 
"""

import h5py
import numpy as np

# new and old hdf5 regression files
fp_old = h5py.File("model_regression_data-original.h5")
fp_new = h5py.File("model_regression_data.h5",'r')

# name of new model with regression data in fp_new but not fp_old
# this will exclude NEW_MODEL from the check
# set this to a nonsenese name if we want to check all models
new_model = "NRHybSur3dq8_CCE"


# Visual check: all enteries are the same except the new one added to fp_new
print("models in new datafile")
for model in fp_new.keys(): # loop over models
    print(model)
print("\n\n---------------\n\n")
print("models in old datafile")
for model in fp_old.keys(): # loop over models
    print(model)


for model in fp_new.keys(): # loop over models
    if model != new_model: # Skip new model we just added
        model_regession_data = fp_new[model]
        for paramter in model_regession_data.keys(): # loop over parameters
            model_regession_data_i = model_regession_data[paramter]
            for data in model_regession_data_i.keys(): # loop over data for each parameter
                if data != "parameter" and data != "time":
                    x = '/'+model+'/'+paramter+'/'+data
                    #print(x)
                    y1 = model_regession_data_i[data][:,:]
                    y2 = fp_old[x][:,:]
                    y = y1 - y2
                    #print( np.max(np.abs(y1)) )
                    if np.max(np.abs(y)) == 0.0:
                        pass
                    else:
                        print("New and old regression data doesn't agree.")
                        print(x)
                        print(y)
