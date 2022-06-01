""" download all models to be tested in test_model_regression.py

This is useful to do when using continuous integration. """

import gwsurrogate as gws
import hashlib, os


def md5(fname):
  """ Compute has from file. code taken from
  https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file"""
  hash_md5 = hashlib.md5()
  with open(fname, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_md5.update(chunk)
  return hash_md5.hexdigest()

models = ['SpEC_q1_10_NoSpin_linear_alt',
'NRHybSur3dq8',
'SpEC_q1_10_NoSpin',
'SpEC_q1_10_NoSpin_linear',
'NRSur7dq4',
'NRHybSur2dq15'
]

for model in models:
  print("Downloading model %s ..."%model)
  gws.catalog.pull(model)
  surr_url = gws.catalog._surrogate_world[model].url
  path_to_model = gws.catalog.download_path()+os.path.basename(surr_url)
  print("md5 Hash of %s is %s"%(model,md5(path_to_model)))
  if not gws.catalog.is_file_recent(path_to_model):
     print("File download failed!") 
     assert(False)