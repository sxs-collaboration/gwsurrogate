"""Discover, download, and verify surrogate models available with gwsurrogate.

Notes
-----
Many surrogate data files are hosted on Zenodo. A new record gives a file a new URL,
even when its contents have not changed. Compare the file's MD5 hash with the value
stored in its ``surrogate_info`` entry to check whether the contents match.

When adding a model that will be available through PyCBC, also update ``setup.py``.

GitHub Actions hashes this file and ``test/download_regression_models.py`` to select
its model cache. See ``.github/workflows/python-app.yml`` for cache selection.
The `pull` function verifies each model against its catalog MD5, including after a cache hit.
"""

from __future__ import division # for python 2


__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@umassd.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Jonathan Blackman, Scott Field, Chad Galley, Kevin Barkett"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
import hashlib
import requests
import shutil
from collections import namedtuple
from datetime import datetime, timezone
from glob import glob
import tarfile
import tempfile
from urllib.parse import urlsplit

### Naming convention: dictionary KEY should match file name KEY.tar.gz ###
surrogate_info = namedtuple('surrogate_info', ['url', 'desc', 'refs', 'md5'])

### dictionary of all known surrogates ###
_surrogate_world = {}

_surrogate_world['EOBNRv2'] = \
  surrogate_info('https://www.dropbox.com/scl/fi/8tyein6dmc1mfzmtbp56b/EOBNRv2.tar.gz?rlkey=3j80zg9vn79744p9w9ttemug2&e=1&st=rrp9ubx2&dl=1',
               ''' Collection of single mode surrogates from mass ratios 1 to 10,
               as long as 190000M and modes (2,1), (2,2), (3,3), (4,4), (5,5). This is not
               a true multi-mode surrogate, and relative time/phase information between the
               modes have not been preserved.''',
               '''http://journals.aps.org/prx/abstract/10.1103/PhysRevX.4.031006''',
               'e8c7ded3b533c7b13df973155c36badb')

_surrogate_world['SpEC_q1_10_NoSpin'] = \
  surrogate_info('https://zenodo.org/record/3348115/files/SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0.h5',
                 '''A multimode surrogate model built from numerical relativity simulations
               performed with SpEC.The surrogate covers mass ratios from 1 to 10, durations
               corresponding to about 15 orbits before merger, and many harmonic modes.''',
                 '''http://arxiv.org/abs/1502.07758''',
                 '4d08862a85437e76a1634dae6d984fdb')

_surrogate_world['SpEC_q1_10_NoSpin_linear'] = \
  surrogate_info('http://www.math.umassd.edu/~sfield/external/surrogates/SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0_FastSplined_WithVandermonde.h5',
                 '''Linear surrogate (using fast splines) version of the SpEC_q1_10_NoSpin.
                 This surrogate is designed to be loaded with the original gws interface.''',
                 '''http://iopscience.iop.org/article/10.1088/1361-6382/aa7649/meta''',
                 '3f8bd987b0473ac068d91b284e7d3d2e')

_surrogate_world['SpEC_q1_10_NoSpin_linear_alt'] = \
  surrogate_info('http://www.math.umassd.edu/~sfield/external/surrogates/SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0_FastSplined_WithVandermonde_NewInterface.h5',
               '''Linear surrogate (using fast splines) version of the SpEC_q1_10_NoSpin.
               This surrogate is designed to be loaded with an alternative (experimental)
               gws interface.''',
               '''http://iopscience.iop.org/article/10.1088/1361-6382/aa7649/meta''',
               '6ae4249bc2c420fa27553d07f4df62df')

_surrogate_world['NRSur4d2s_TDROM_grid12'] = \
  surrogate_info('https://zenodo.org/record/3348115/files/NRSur4d2s_TDROM_grid12.h5',
               '''Fast time-domain surrogate model for binary black hole mergers where the
               black holes may be spinning, but the spins are restricted to a parameter
               subspace which includes some but not all precessing configurations.
               NRSur4d2s_TDROM_grid12.h5 is built from the underlying (slower) NRSur4d2s
               time-domain model. Additional tools for acceleration use splines (see the
               frequency-domain discussion of the refs)''',
               '''https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.104023''',
               '44fba833b6b3a0f269fc788df181dfd4')

_surrogate_world['NRSur4d2s_FDROM_grid12'] = \
  surrogate_info('https://zenodo.org/record/3348115/files/NRSur4d2s_FDROM_grid12.h5',
               '''Fast frequency-domain surrogate model for binary black hole mergers where
               the black holes may be spinning, but the spins are restricted to a parameter
               subspace which includes some but not all precessing configurations.''',
               '''https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.104023''',
               'ec8bf594c36ba76e1198dfc01ee1861f')


_surrogate_world['NRHybSur3dq8'] = \
  surrogate_info(\
  'https://zenodo.org/record/3348115/files/NRHybSur3dq8.h5',
  '''Surrogate model for aligned-spin binary black holes with mass ratios q<=8
  and spin magnitudes <=0.8. The model is trained on NR waveforms that have been
  hybridized using EOB/PN and spans the entire LIGO frequency band. This model
  is  presented in Varma et al. 2018, arxiv:1812.07865. Available modes are
  [(2,2), (2,1), (2,0), (3,3), (3,2), (3,1), (3,0), (4,4), (4,3), (4,2) and
  (5,5)]. The m<0 modes are deduced from the m>0 modes.''',
  '''https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.064045''',
  'b42cd577f497b1db3da14f1e4ee0ccd1')

_surrogate_world['NRHybSur3dq8_CCE'] = \
  surrogate_info(\
  'https://zenodo.org/record/8246990/files/NRHybSur3dq8_CCE.h5',
  '''CCE Surrogate model for aligned-spin binary black holes with mass ratios q<=8
  and spin magnitudes <=0.8. The model is trained on NR (CCE) waveforms that have been
  hybridized using EOB/PN and spans the entire LIGO frequency band.
  NRHybSur3dq8_CCE captures memory effects while NRHybSur3dq8 does not. This model
  is  presented in arXiv:2306.03148. Available modes are
  [(2,2), (2,1), (2,0), (3,3), (3,2), (3,0), (4,4), (4,3), (4,0), and (5,5)]. 
  The m<0 modes are deduced from the m>0 modes.''',
  '''https://arxiv.org/abs/2306.03148''',
  '58fa10c2b35d37d0269f9e4b7157c23a')

_surrogate_world['NRHybSur2dq15'] = \
  surrogate_info(\
  'https://zenodo.org/record/6726994/files/NRHybSur2dq15.h5',
  '''Surrogate model for aligned-spin binary black holes with mass ratios q<=15,
  primary spin magnitudes <=0.5, and zero spin on secondary. 
  The model is trained on NR waveforms that have been
  hybridized using EOB/PN and spans the entire LIGO frequency band. This model
  is  presented in arxiv:2203.10109. Available modes are
  [(2,2), (2,1), (3,3), (4,4), and (5,5)]. The m<0 modes are deduced from the m>0 modes.''',
  '''https://arxiv.org/abs/2203.10109''',
  '140af07f2864e4e513eff648aaf8a7de')

_surrogate_world['NRSur7dq4'] = \
  surrogate_info(\
  'https://zenodo.org/record/3348115/files/NRSur7dq4.h5',
  '''Surrogate model for precessing binary black holes with mass ratios q<=4
  and spin magnitudes <=0.8. This model is presented in Varma et al. 2019,
  arxiv:1905.09300. All ell<=4 modes are included. The spin and frame dynamics
  are also modeled.''',
  '''https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.1.033015''',
  '8e033ba4e4da1534b3738ae51549fb98')

_surrogate_world['NRSur7dq4v2'] = \
 surrogate_info(\
   'https://zenodo.org/records/22257361/files/NRSur7dq4v2.h5',
   '''Surrogate model with improved ringdown for precessing binary black holes
     with mass ratio q<=4 and spin magnitudes <=0.8. All ell<=5 modes are
     included. The spin and frame dynamics are also modeled.''',
     '''arXiv:2609.07873''',
     '2bef4cfdb12d73904bd727015bef629c')

_surrogate_world['SEOBNRv4PHMSur'] = \
  surrogate_info(\
  'https://zenodo.org/records/13376190/files/SEOBNRv4PHMSur.h5',
  '''Surrogate model for the time domain precessing EOB waveform model
  SEOBNRv4PHM (). The model is valid for mass ratio <= 20 and spin
  magnitudes upto 0.8. Extrapolation for spins works reasonably well till
  0.9 and maybe till 0.95 for q < 5. The model has (2,2), (2,1), (3,3),
  (4,4) and (5,5) modes in coorbital frame. So one can choose ellMax <=5
  for inertial frame waveforms. Surrogate is 5000M long.''',
  '''https://arxiv.org/abs/2203.00381''',
  '2ce450b06ca29d24d538dc86e81a31d4')

_surrogate_world['NRHybSur3dq8Tidal'] = \
  surrogate_info(\
  'https://zenodo.org/record/3348115/files/NRHybSur3dq8.h5',
  '''Surrogate model 'NRHybSur3dq8' modified by splicing in PN tidal
  approximants for aligned-spin binary neutron stars/black hole-neutron star
  systems with mass ratio q<=8 and spin magnitudes <=.7; please see the
  NRHybSur3dq8Tidal class doctring for why these restrictions are smaller
  than the NRHybSur3dq8 model. The model is spliced using the
  TaylorT2 expansion and spans the entire LIGO frequency band. This
  model is presented in Barkett et al. 2019, arxiv:xxxx.xxxxx #FIXME. Available
  modes are [(2,2), (2,1), (2,0), (3,3), (3,2), (3,1), (3,0), (4,4), (4,3),
  (4,2) and (5,5)]. The m<0 modes are deduced from the m>0 modes. The same
  hdf5 file is used for both NRHybSur3dq8Tidal and NRHybSur3dq8 models. ''',
  '''https://arxiv.org/abs/1911.10440''',
  'b42cd577f497b1db3da14f1e4ee0ccd1')

_surrogate_world['EMRISur1dq1e4'] = \
  surrogate_info(\
  'https://zenodo.org/record/7125742/files/EMRISur1dq1e4.h5',
  '''Surrogate model 'EMRISur1dq1e4' for non-spinning black hole binary
  systems with mass-ratios varying from 3 to 10000. This surrogate model
  is trained on waveform data generated by point-particle black hole
  perturbation theory (ppBHPT), with the total mass rescaling parameter tuned
  to NR simulations according to the paper's Eq. 4. Note that this rescaling
  is applied in EvaluateSingleModeSurrogate's call method, and to generate
  point-particle perturbation theory waveforms set alpha_emri = 1.
  Available modes are [(2,2), (2,1), (3,3), (3,2), (3,1), (4,4), (4,3),
  (4,2), (5,5), (5,4), (5,3)]. The m<0 modes are deduced from the m>0 modes.
  Model details can be found in Rifat et al. 2019, arXiv:1910.10473. NOTE:
  the datasets in this hdf5 file are 32-bit (single) precision. Some are up-cast
  to double in SurrogateIO. This model has been superseded by BHPTNRSur1dq1e4.''',
  '''https://arxiv.org/abs/1910.10473''',
  'd145958484738e0c7292e084a66a96fa')

_surrogate_world['BHPTNRSur1dq1e4'] = \
  surrogate_info(\
  'https://zenodo.org/record/7125742/files/BHPTNRSur1dq1e4.h5',
  '''Surrogate model 'BHPTNRSur1dq1e4' for non-spinning black hole binary
  systems with mass-ratios varying from 2.5 to 10000. This surrogate model
  is trained on waveform data generated by point-particle black hole
  perturbation theory (ppBHPT), and tuned to NR simulations in the comparable 
  mass ratio regime (q=3 to q=10). Available modes are: (2,1),(2,2),(3,1),(3,2),
  (3,3),(4,2),(4,3),(4,4),(5,3),(5,4),(5,5),(6,4),(6,5),(6,6),(7,5),(7,6),(7,7),
  (8,6),(8,7),(8,8),(9,7),(9,8),(9,9),(10,8),(10,9)]. The m<0 modes are deduced 
  from the m>0 modes. Model details can be found in Islam et al. 2022, arXiv:2204.01972.''',
  '''https://arxiv.org/abs/2204.01972''',
  '58a3a75e8fd18786ecc88cf98f694d4a')

def _md5(filename):
  """Compute a file's MD5 hash in chunks.

  Parameters
  ----------
  filename : str or os.PathLike
      Path to the file to read in binary mode.

  Returns
  -------
  str
      Hexadecimal MD5 digest of the file contents.

  Raises
  ------
  OSError
      If the file cannot be opened or read.
  """

  hash_md5 = hashlib.md5()
  with open(filename, "rb") as f:
    for chunk in iter(lambda: f.read(1024*1024), b""):
      hash_md5.update(chunk)
  return hash_md5.hexdigest()

def is_file_recent(filename):
  """Check whether a local model file matches its catalog checksum.

  Parameters
  ----------
  filename : str or os.PathLike
      Path to a local data file or archive whose filename appears in the catalog.

  Returns
  -------
  bool
      True if the file's MD5 matches the catalog checksum, otherwise False.

  Raises
  ------
  ValueError
      If the filename does not match any model in the catalog.
  OSError
      If the file cannot be opened or read.

  See Also
  --------
  pull : Download or reuse a verified model file.
  get_modelID_from_filename : Find catalog model IDs associated with a filename.

  Notes
  -----
  The comparison uses the installed catalog; no request is made to the download host.
  If multiple models share the filename, the first matching catalog entry is used.
  """

  names = get_modelID_from_filename(filename)
  if not names:
    raise ValueError("No surrogate package matches %s"%filename)
  return _md5(filename) == _surrogate_world[names[0]].md5

def download_path():
  """Return the default directory for downloaded surrogate models.

  Returns
  -------
  str
      Path to the ``surrogate_downloads`` directory inside the installed gwsurrogate
      package, including a trailing slash. The directory is not created by this function.
  """

  import gwsurrogate
  import os
  gws_path = os.path.dirname(gwsurrogate.__file__)
  return gws_path+'/surrogate_downloads/'

def list(verbose=False):
  """Print the catalog of surrogate models available for download.

  Parameters
  ----------
  verbose : bool, optional
      If True, also print each model's URL, MD5 checksum, description, and references.
      The default is False, which prints only model IDs.

  Returns
  -------
  None
      Model information is printed to standard output.
  """

  for surr_key in _surrogate_world.keys():
    print(surr_key)
    if verbose:
        print('  url: '+_surrogate_world[surr_key].url)
        print('  md5 hash: %s'%str(_surrogate_world[surr_key].md5))
        print("  Description: " + _surrogate_world[surr_key].desc)
        print("  References: "+_surrogate_world[surr_key].refs+'\n')


def get_modelID_from_filename(filename):
  """Find the catalog model IDs associated with a filename.

  Parameters
  ----------
  filename : str or os.PathLike
      Data file or archive name, optionally including its directory path.
      The file does not need to exist.

  Returns
  -------
  modelIDs : list of str
      All matching model IDs in catalog order, including models that share a data file.
      An empty list is returned if no model matches.

  Notes
  -----
  Only the basename is compared with the filename in each catalog URL's path.
  URL query parameters and fragments do not participate in the comparison.
  """

  file_without_path = os.path.basename(filename)
  modelIDs = []
  for modelID in _surrogate_world.keys():
    url = _surrogate_world[modelID].url
    if os.path.basename(urlsplit(url).path) == file_without_path:
      modelIDs.append(modelID)
  return modelIDs



def _unzip(surr_name,sdir=download_path()):
  """Extract a surrogate archive while retaining the original download.

  Parameters
  ----------
  surr_name : str
      Name of the ``.tar.gz`` archive in `sdir`.
  sdir : str or os.PathLike, optional
      Directory containing the archive and receiving the extracted files.
      Defaults to `download_path`.

  Returns
  -------
  str
      Absolute path to the surrogate directory, named after the archive without
      the ``.tar.gz`` suffix.

  Raises
  ------
  tarfile.TarError
      If the archive cannot be read or extraction fails with a tarfile error.
  OSError
      If the archive cannot be opened or extracted files cannot be written.

  Notes
  -----
  The archive is retained for later checksum verification. Extraction does not
  change the working directory. The caller is responsible for verifying the archive.
  """

  sdir = os.path.abspath(sdir)
  with tarfile.open(os.path.join(sdir, surr_name), "r:gz") as t:
    t.extractall(path=sdir)

  return os.path.join(sdir, surr_name[:-len('.tar.gz')])

def pull(surr_name,sdir=download_path(),force=False):
  """Download or reuse a verified local surrogate model.

  Parameters
  ----------
  surr_name : str
      Model ID in the catalog. Use `list` to print the available IDs.
  sdir : str or os.PathLike, optional
      Directory for downloaded models. Defaults to `download_path` and is created
      if needed when downloading a model.
  force : bool, optional
      If True, download the model even when the existing file matches its catalog
      checksum. The new download must still pass verification. The default is False.

  Returns
  -------
  str
      Absolute path to the verified data file, or to the extracted surrogate
      directory for a ``.tar.gz`` model.

  Raises
  ------
  ValueError
      If the model ID is unknown or the downloaded file's MD5 differs from the catalog.
  requests.exceptions.RequestException
      If an HTTP request or transfer fails.
  OSError
      If a local file cannot be read, written, backed up, or replaced.
  tarfile.TarError
      If the downloaded archive cannot be read or extracted by tarfile.

  See Also
  --------
  list : Print available model IDs and metadata.
  download_path : Get the default model download directory.
  is_file_recent : Check an existing file against its catalog checksum.

  Notes
  -----
  Existing files are reused when their MD5 matches the installed catalog, unless
  `force` is True. Downloaded data is written to a temporary file and verified before
  installation. An existing file is copied to a timestamped backup in ``sdir/backup``
  before replacement. Failed downloads leave existing files untouched and temporary
  downloads are removed.

  For ``.tar.gz`` models, the checksum applies to the archive. The verified archive
  is retained and extracted on every call, including reuse, without changing the
  working directory.

  Examples
  --------
  Download a model or reuse its verified local copy:

  >>> from gwsurrogate import catalog
  >>> path = catalog.pull('NRHybSur3dq8')  # doctest: +SKIP

  Download a fresh copy into a chosen directory:

  >>> path = catalog.pull('NRHybSur3dq8', sdir='models', force=True)  # doctest: +SKIP
  """

  if surr_name not in _surrogate_world:
    raise ValueError("No surrogate package exists")

  info = _surrogate_world[surr_name]
  sdir = os.path.abspath(sdir)
  fname = os.path.basename(urlsplit(info.url).path)
  output_path = os.path.join(sdir, fname)

  if not force and os.path.isfile(output_path) and _md5(output_path) == info.md5:
    print("Reusing model %s at %s (MD5 matches catalog)."%(surr_name, output_path))
  else:
    print("Downloading model %s ..."%surr_name)
    os.makedirs(sdir, exist_ok=True)
    temp_path = None
    try:
      # Keep incomplete or incorrect downloads separate from the installed file.
      hash_md5 = hashlib.md5()
      with tempfile.NamedTemporaryFile(dir=sdir, prefix=fname+'.', suffix='.part', delete=False) as f:
        temp_path = f.name
        with requests.get(info.url, stream=True) as r:
          r.raise_for_status()
          for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
            hash_md5.update(chunk)

      file_hash = hash_md5.hexdigest()
      if file_hash != info.md5:
        raise ValueError("MD5 mismatch for %s: expected %s, downloaded %s"%(surr_name, info.md5, file_hash))

      # Preserve the existing file until its verified replacement is installed.
      if os.path.isfile(output_path):
        timestamp = datetime.now(timezone.utc).strftime("%Y%b%d_%Hh%Mm%Ss_%f")
        backup_fname = '%s_%s'%(timestamp, fname)
        backup_dir = os.path.join(sdir, 'backup')
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, backup_fname)
        print('\n%s file exists, backing up to %s.'%(fname, backup_path))
        shutil.copy2(output_path, backup_path)
        number_of_backup_files = glob(os.path.join(backup_dir, '*_'+fname))
        if len(number_of_backup_files) > 5:
          print('There are a lot of backup files in %s, consider removing some.'%backup_dir)

      os.replace(temp_path, output_path)
      print("Downloaded model %s to %s (MD5 matches catalog)."%(surr_name, output_path))
    finally:
      if temp_path is not None and os.path.exists(temp_path):
        os.remove(temp_path)

  if fname.endswith('.tar.gz'):
    return _unzip(fname,sdir)
  return output_path
