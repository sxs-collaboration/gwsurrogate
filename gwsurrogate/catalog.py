""" Surrogate download tool """

from __future__ import division # for python 2


__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@umassd.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Jonathan Blackman, Scott Field, Chad Galley"

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
surrogate_info = namedtuple('surrogate_info', ['url', 'desc', 'refs'])

### dictionary of all known surrogates ###
_surrogate_world = {}

_surrogate_world['EOBNRv2'] = \
  surrogate_info('https://www.dropbox.com/s/uyliuy37uczu3ug/EOBNRv2.tar.gz',
                 ''' Collection of single mode surrogates from mass ratios 1 to 10,
               as long as 190000M and modes (2,1), (2,2), (3,3), (4,4), (5,5). This is not
               a true multi-mode surrogate, and relative time/phase information between the
               modes have not been preserved.''',
                 '''http://journals.aps.org/prx/abstract/10.1103/PhysRevX.4.031006''')

_surrogate_world['SpEC_q1_10_NoSpin'] = \
  surrogate_info('https://zenodo.org/record/1215824/files/SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0.h5',
                 '''A multimode surrogate model built from numerical relativity simulations
               performed with SpEC.The surrogate covers mass ratios from 1 to 10, durations
               corresponding to about 15 orbits before merger, and many harmonic modes.''',
                 '''http://arxiv.org/abs/1502.07758''')

_surrogate_world['SpEC_q1_10_NoSpin_linear'] = \
  surrogate_info('http://www.math.umassd.edu/~sfield/external/surrogates/SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0_FastSplined_WithVandermonde.h5',
                 '''Linear surrogate (using fast splines) version of the SpEC_q1_10_NoSpin.
                 This surrogate is designed to be loaded with the original gws interface.''',
                 '''http://iopscience.iop.org/article/10.1088/1361-6382/aa7649/meta''')

_surrogate_world['SpEC_q1_10_NoSpin_linear_alt'] = \
  surrogate_info('http://www.math.umassd.edu/~sfield/external/surrogates/SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0_FastSplined_WithVandermonde_NewInterface.h5',
                 '''Linear surrogate (using fast splines) version of the SpEC_q1_10_NoSpin.
               This surrogate is designed to be loaded with an alternative (experimental)
               gws interface.''',
                 '''http://iopscience.iop.org/article/10.1088/1361-6382/aa7649/meta''')

_surrogate_world['NRSur4d2s_TDROM_grid12'] = \
  surrogate_info('https://zenodo.org/record/1215824/files/NRSur4d2s_TDROM_grid12.h5',
                 '''Fast time-domain surrogate model for binary black hole mergers where the
               black holes may be spinning, but the spins are restricted to a parameter
               subspace which includes some but not all precessing configurations.
               NRSur4d2s_TDROM_grid12.h5 is built from the underlying (slower) NRSur4d2s
               time-domain model. Additional tools for acceleration use splines (see the
               frequency-domain discussion of the refs)''',
                 '''https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.104023''')

_surrogate_world['NRSur4d2s_FDROM_grid12'] = \
  surrogate_info('https://zenodo.org/record/1215824/files/NRSur4d2s_FDROM_grid12.h5',
                 '''Fast frequency-domain surrogate model for binary black hole mergers where
               the black holes may be spinning, but the spins are restricted to a parameter
               subspace which includes some but not all precessing configurations.''',
                 '''https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.104023''')


_surrogate_world['NRHybSur3dq8'] = \
  surrogate_info(\
  'https://www.dropbox.com/s/v10upxsdcdq5nim/NRHybSur3dq8.h5',
  '''Surrogate model for aligned-spin binary black holes with mass ratios q<=8
  and spin magnitudes <=0.8. The model is trained on NR waveforms that have been
  hybridized using EOB/PN and spans the entire LIGO frequency band. This model
  is  presented in Varma et al. 2018, in prep. Available modes are
  [(2,2), (2,1), (2,0), (3,3), (3,2), (3,1), (3,0), (4,4), (4,3), (4,2) and
  (5,5)]. The m<0 modes are deduced from the m>0 modes.''',
  ''' ''')


def download_path():
  """return the default path for downloaded surrogates"""

  import gwsurrogate
  import os
  gws_path = os.path.dirname(gwsurrogate.__file__)
  return gws_path+'/../surrogate_downloads/'

def list():
  """show all known surrogates available for download"""

  for surr_key in _surrogate_world.keys():
    print(surr_key+'...')
    print('  url: '+_surrogate_world[surr_key].url)
    print("  Description: " + _surrogate_world[surr_key].desc)
    print("  References: "+_surrogate_world[surr_key].refs+'\n')

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

    os.system('wget -q --show-progress --directory-prefix='+sdir+' '+surr_url)
  else:
    raise ValueError("No surrogate package exits")

  # deduce the surrogate file name and extension type
  # one can directly load a surrogate from surr_path
  file_name = surr_url.split('/')[-1]
  if file_name.split('.')[1] == 'tar': # assumed to be *.h5 or *.tar.gz
    surr_path = _unzip(file_name,sdir)
  else:
    surr_path = sdir+file_name

  return surr_path

