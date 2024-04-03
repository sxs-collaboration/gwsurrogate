""" Surrogate catalog. Information on all models available with gwsurrogate.

NOTES: 

(*) Many surrogate data files come from zenodo. When a new record is
generated, a new URL will be too. However, the file contents may
not be changed despite the new record. File similarity can be
checked by computing the md5 hash and comparing with the value stored
in the surrogate_info tuple. 

(*) If your model will be available to pycbc, please also edit
setup.py.
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
from collections import namedtuple
from time import gmtime, strftime
from glob import glob

### Naming convention: dictionary KEY should match file name KEY.tar.gz ###
surrogate_info = namedtuple('surrogate_info', ['url', 'desc', 'refs', 'md5', 'bib'])

### dictionary of all known surrogates ###
_surrogate_world = {}

_surrogate_world['EOBNRv2'] = \
  surrogate_info('https://www.dropbox.com/s/uyliuy37uczu3ug/EOBNRv2.tar.gz',
                 '''Collection of single mode surrogates from mass ratios 1 to 10,
               as long as 190000M and modes (2,1), (2,2), (3,3), (4,4), (5,5). This is not
               a true multi-mode surrogate, and relative time/phase information between the
               modes have not been preserved.''',
                 '''http://journals.aps.org/prx/abstract/10.1103/PhysRevX.4.031006''',
               None,
                 '''
  @article{Field:2013cfa,
      author = "Field, Scott E. and Galley, Chad R. and Hesthaven, Jan S. and Kaye, Jason and Tiglio, Manuel",
      title = "{Fast prediction and evaluation of gravitational waveforms using surrogate models}",
      eprint = "1308.3565",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      doi = "10.1103/PhysRevX.4.031006",
      journal = "Phys. Rev. X",
      volume = "4",
      number = "3",
      pages = "031006",
      year = "2014"
  }''')

_surrogate_world['SpEC_q1_10_NoSpin'] = \
  surrogate_info('https://zenodo.org/record/3348115/files/SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0.h5',
                 '''A multimode surrogate model built from numerical relativity simulations
               performed with SpEC.The surrogate covers mass ratios from 1 to 10, durations
               corresponding to about 15 orbits before merger, and many harmonic modes.''',
                 '''http://arxiv.org/abs/1502.07758''',
                 '4d08862a85437e76a1634dae6d984fdb',
                 '''
  @article{Blackman:2015pia,
      author = "Blackman, Jonathan and Field, Scott E. and Galley, Chad R. and Szil\'agyi, B\'ela and Scheel, Mark A. and Tiglio, Manuel and Hemberger, Daniel A.",
      title = "{Fast and Accurate Prediction of Numerical Relativity Waveforms from Binary Black Hole Coalescences Using Surrogate Models}",
      eprint = "1502.07758",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      doi = "10.1103/PhysRevLett.115.121102",
      journal = "Phys. Rev. Lett.",
      volume = "115",
      number = "12",
      pages = "121102",
      year = "2015"
  }''')

_surrogate_world['SpEC_q1_10_NoSpin_linear'] = \
  surrogate_info('http://www.math.umassd.edu/~sfield/external/surrogates/SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0_FastSplined_WithVandermonde.h5',
                 '''Linear surrogate (using fast splines) version of the SpEC_q1_10_NoSpin.
                 This surrogate is designed to be loaded with the original gws interface.''',
                 '''https://iopscience.iop.org/article/10.1088/1361-6382/aa7649''',
                 '3f8bd987b0473ac068d91b284e7d3d2e',
                 '''
  @article{OShaughnessy:2017tak,
      author = "O'Shaughnessy, Richard and Blackman, Jonathan and Field, Scott E.",
      title = "{An architecture for efficient gravitational wave parameter estimation with multimodal linear surrogate models}",
      eprint = "1701.01137",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      reportNumber = "LIGO-P1600309",
      doi = "10.1088/1361-6382/aa7649",
      journal = "Class. Quant. Grav.",
      volume = "34",
      number = "14",
      pages = "144002",
      year = "2017"
  }''')

_surrogate_world['SpEC_q1_10_NoSpin_linear_alt'] = \
  surrogate_info('http://www.math.umassd.edu/~sfield/external/surrogates/SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0_FastSplined_WithVandermonde_NewInterface.h5',
               '''Linear surrogate (using fast splines) version of the SpEC_q1_10_NoSpin.
               This surrogate is designed to be loaded with an alternative (experimental)
               gws interface.''',
               '''https://iopscience.iop.org/article/10.1088/1361-6382/aa7649''',
               '6ae4249bc2c420fa27553d07f4df62df',
               '''
  @article{OShaughnessy:2017tak,
      author = "O'Shaughnessy, Richard and Blackman, Jonathan and Field, Scott E.",
      title = "{An architecture for efficient gravitational wave parameter estimation with multimodal linear surrogate models}",
      eprint = "1701.01137",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      reportNumber = "LIGO-P1600309",
      doi = "10.1088/1361-6382/aa7649",
      journal = "Class. Quant. Grav.",
      volume = "34",
      number = "14",
      pages = "144002",
      year = "2017"
  }''')

_surrogate_world['NRSur4d2s_TDROM_grid12'] = \
  surrogate_info('https://zenodo.org/record/3348115/files/NRSur4d2s_TDROM_grid12.h5',
               '''Fast time-domain surrogate model for binary black hole mergers where the
               black holes may be spinning, but the spins are restricted to a parameter
               subspace which includes some but not all precessing configurations.
               NRSur4d2s_TDROM_grid12.h5 is built from the underlying (slower) NRSur4d2s
               time-domain model. Additional tools for acceleration use splines (see the
               frequency-domain discussion of the refs)''',
               '''https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.104023''',
               '44fba833b6b3a0f269fc788df181dfd4',
               '''
  @article{Blackman:2017dfb,
      author = "Blackman, Jonathan and Field, Scott E. and Scheel, Mark A. and Galley, Chad R. and Hemberger, Daniel A. and Schmidt, Patricia and Smith, Rory",
      title = "{A Surrogate Model of Gravitational Waveforms from Numerical Relativity Simulations of Precessing Binary Black Hole Mergers}",
      eprint = "1701.00550",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      doi = "10.1103/PhysRevD.95.104023",
      journal = "Phys. Rev. D",
      volume = "95",
      number = "10",
      pages = "104023",
      year = "2017"
  }''')

_surrogate_world['NRSur4d2s_FDROM_grid12'] = \
  surrogate_info('https://zenodo.org/record/3348115/files/NRSur4d2s_FDROM_grid12.h5',
               '''Fast frequency-domain surrogate model for binary black hole mergers where
               the black holes may be spinning, but the spins are restricted to a parameter
               subspace which includes some but not all precessing configurations.''',
               '''https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.104023''',
               'ec8bf594c36ba76e1198dfc01ee1861f',
               '''
  @article{Blackman:2017dfb,
      author = "Blackman, Jonathan and Field, Scott E. and Scheel, Mark A. and Galley, Chad R. and Hemberger, Daniel A. and Schmidt, Patricia and Smith, Rory",
      title = "{A Surrogate Model of Gravitational Waveforms from Numerical Relativity Simulations of Precessing Binary Black Hole Mergers}",
      eprint = "1701.00550",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      doi = "10.1103/PhysRevD.95.104023",
      journal = "Phys. Rev. D",
      volume = "95",
      number = "10",
      pages = "104023",
      year = "2017"
  }''')


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
  'b42cd577f497b1db3da14f1e4ee0ccd1',
  '''
  @article{Varma:2018mmi,
      author = "Varma, Vijay and Field, Scott E. and Scheel, Mark A. and Blackman, Jonathan and Kidder, Lawrence E. and Pfeiffer, Harald P.",
      title = "{Surrogate model of hybridized numerical relativity binary black hole waveforms}",
      eprint = "1812.07865",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      doi = "10.1103/PhysRevD.99.064045",
      journal = "Phys. Rev. D",
      volume = "99",
      number = "6",
      pages = "064045",
      year = "2019"
  }''')

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
  '58fa10c2b35d37d0269f9e4b7157c23a',
  '''
  @article{Yoo:2023spi,
      author = "Yoo, Jooheon and others",
      title = "{Numerical relativity surrogate model with memory effects and post-Newtonian hybridization}",
      eprint = "2306.03148",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      doi = "10.1103/PhysRevD.108.064027",
      journal = "Phys. Rev. D",
      volume = "108",
      number = "6",
      pages = "064027",
      year = "2023"
  }''')

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
  '140af07f2864e4e513eff648aaf8a7de',
  '''
  @article{Yoo:2022erv,
      author = "Yoo, Jooheon and Varma, Vijay and Giesler, Matthew and Scheel, Mark A. and Haster, Carl-Johan and Pfeiffer, Harald P. and Kidder, Lawrence E. and Boyle, Michael",
      title = "{Targeted large mass ratio numerical relativity surrogate waveform model for GW190814}",
      eprint = "2203.10109",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      doi = "10.1103/PhysRevD.106.044001",
      journal = "Phys. Rev. D",
      volume = "106",
      number = "4",
      pages = "044001",
      year = "2022"
  }''')

_surrogate_world['NRSur7dq4'] = \
  surrogate_info(\
  'https://zenodo.org/record/3348115/files/NRSur7dq4.h5',
  '''Surrogate model for precessing binary black holes with mass ratios q<=4
  and spin magnitudes <=0.8. This model is presented in Varma et al. 2019,
  arxiv:1905.09300. All ell<=4 modes are included. The spin and frame dynamics
  are also modeled.''',
  '''https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.1.033015''',
  '8e033ba4e4da1534b3738ae51549fb98',
  '''
  @article{Varma:2019csw,
      author = "Varma, Vijay and Field, Scott E. and Scheel, Mark A. and Blackman, Jonathan and Gerosa, Davide and Stein, Leo C. and Kidder, Lawrence E. and Pfeiffer, Harald P.",
      title = "{Surrogate models for precessing binary black hole simulations with unequal masses}",
      eprint = "1905.09300",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      doi = "10.1103/PhysRevResearch.1.033015",
      journal = "Phys. Rev. Research.",
      volume = "1",
      pages = "033015",
      year = "2019"
  }''')

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
  'b42cd577f497b1db3da14f1e4ee0ccd1',
  '''
  @article{Barkett:2019tus,
      author = "Barkett, Kevin and Chen, Yanbei and Scheel, Mark A. and Varma, Vijay",
      title = "{Gravitational waveforms of binary neutron star inspirals using post-Newtonian tidal splicing}",
      eprint = "1911.10440",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      doi = "10.1103/PhysRevD.102.024031",
      journal = "Phys. Rev. D",
      volume = "102",
      number = "2",
      pages = "024031",
      year = "2020"
  }''')

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
  'd145958484738e0c7292e084a66a96fa',
  '''
  @article{Rifat:2019ltp,
      author = "Rifat, Nur E. M. and Field, Scott E. and Khanna, Gaurav and Varma, Vijay",
      title = "{Surrogate model for gravitational wave signals from comparable and large-mass-ratio black hole binaries}",
      eprint = "1910.10473",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      doi = "10.1103/PhysRevD.101.081502",
      journal = "Phys. Rev. D",
      volume = "101",
      number = "8",
      pages = "081502",
      year = "2020"
  }''')

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
  '58a3a75e8fd18786ecc88cf98f694d4a',
  '''
  @article{Islam:2022laz,
      author = "Islam, Tousif and Field, Scott E. and Hughes, Scott A. and Khanna, Gaurav and Varma, Vijay and Giesler, Matthew and Scheel, Mark A. and Kidder, Lawrence E. and Pfeiffer, Harald P.",
      title = "{Surrogate model for gravitational wave signals from nonspinning, comparable-to large-mass-ratio black hole binaries built on black hole perturbation theory waveforms calibrated to numerical relativity}",
      eprint = "2204.01972",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      doi = "10.1103/PhysRevD.106.104025",
      journal = "Phys. Rev. D",
      volume = "106",
      number = "10",
      pages = "104025",
      year = "2022"
  }''')

# TODO: test function, and then use it whenever a file is loaded
def is_file_recent(filename):
  """ Check local hdf5 file's hash against most recent one on Zenodo. """

  import hashlib

  def md5(fname):
    """ Compute has from file. code taken from
    https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file"""
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
      for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()

  file_hash = md5(filename)

  names = get_modelID_from_filename(filename)
  modelID = names[0]

  zenodo_current_hash = _surrogate_world[modelID].md5

  if file_hash != zenodo_current_hash:
    return False
    #raise AttributeError("%s out of date.\n Please download the current version"%filename)
  else:
    return True

def download_path():
  """return the default path for downloaded surrogates"""

  import gwsurrogate
  import os
  gws_path = os.path.dirname(gwsurrogate.__file__)
  return gws_path+'/surrogate_downloads/'

def list(verbose=False):
  """show all known surrogates available for download"""

  for surr_key in _surrogate_world.keys():
    print(surr_key)
    if verbose:
        print('  url: '+_surrogate_world[surr_key].url)
        print('  md5 hash: %s'%str(_surrogate_world[surr_key].md5))
        print("  Description: " + _surrogate_world[surr_key].desc)
        print("  References: "+_surrogate_world[surr_key].refs)
        print("  Cite: \n"+_surrogate_world[surr_key].bib+'\n\n')


def get_modelID_from_filename(filename):
  """ From the model's filename (which could be a path),
  return the model's unique ID as a list.

  If multiple models have the same datafile, all matching model ID tags
  are returned. If no match if found, and empty list is returned. """

  file_without_path = filename.split('/')[-1]
  modelIDs = []
  for modelID in _surrogate_world.keys():
    url = _surrogate_world[modelID].url
    if url.find(file_without_path) >=0:
      modelIDs.append(modelID)
  return modelIDs



def _unzip(surr_name,sdir=download_path()):
  """unzip a tar.gz surrogate and remove the tar.gz file"""

  os.chdir(sdir)
  os.system('tar -xvzf '+surr_name)
  os.remove(surr_name)

  return sdir+surr_name.split('.')[0]

def pull(surr_name,sdir=download_path()):
  """pass a valid surr_name from the repo list and download location sdir.
     The default path is used if no location supplied. tar.gz surrogates
     are automatically unziped. The new surrogate path is returned."""

  sdir = os.path.abspath(sdir)
  if surr_name in _surrogate_world:
    surr_url = _surrogate_world[surr_name].url
    fname = os.path.basename(surr_url)

    # If file already exists, move it to backup dir with time stamp
    if os.path.isfile('%s/%s'%(sdir, fname)):
        timestamp=strftime("%Y%b%d_%Hh:%Mm:%Ss", gmtime())
        backup_fname = '%s_%s'%(timestamp, fname)
        backup_dir = '%s/backup'%(sdir)
        os.system('mkdir -p %s'%backup_dir)
        print('\n%s file exits, moving to %s/%s.'%(fname, backup_dir, \
            backup_fname))
        os.system('mv %s/%s %s/%s'%(sdir, fname, backup_dir, backup_fname))
        number_of_backup_files = glob('%s/*_%s'%(backup_dir, fname))
        if len(number_of_backup_files) > 5:
            print('There are a lot of backup files in %s, consider removing'
                ' some.'%backup_dir)

    os.system('wget -q --directory-prefix='+sdir+' '+surr_url)
  else:
    raise ValueError("No surrogate package exits")

  # deduce the surrogate file name and extension type
  # one can directly load a surrogate from surr_path
  file_name = surr_url.split('/')[-1]
  if file_name.split('.')[1] == 'tar': # assumed to be *.h5 or *.tar.gz
    surr_path = _unzip(file_name,sdir)
  else:
    surr_path = sdir+'/'+file_name

  return surr_path


