""" Surrogate download tool """

from __future__ import division

__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@umd.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Scott Field, Chad Galley"

__license__ = """
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see U{http://www.gnu.org/licenses/}.
"""

import os

### Naming convention: dictionary KEY (e.g. EOBNRv2) should match file KEY.tar.gz (e.g. EOBNRv2.tar.gz) ###

### dictionary of all known surrogates -- download location ###
_surrogate_world = {'EOBNRv2':'https://www.dropbox.com/s/4dpiw0zzh1wddmq/EOBNRv2.tar.gz'}

### dictionary of all known surrogates -- information ###
_surrogate_world_info = {'EOBNRv2':'add info here'}

def download_path():
	'''return the default path for downloaded surrogates'''

	import gwsurrogate
	import os
	gws_path = os.path.dirname(gwsurrogate.__file__)
	return gws_path+'/../surrogate_downloads/'

def list():
	'''show all known surrogates available for download'''

	for surr_id, surr_url in _surrogate_world.iteritems():
		print surr_id+' is located at '+surr_url

def get(surr_name,sdir=None):
	'''pass a valid surr_name from the repo list and download location. The default path is used if no location supplied'''

	if( sdir is None):
		download_to = download_path()
	else:
		download_to = sdir

	if _surrogate_world.has_key(surr_name):
		os.system('wget --directory-prefix='+download_to+' '+_surrogate_world[surr_name])
	else:
		raise ValueError("No surrogate package exits")

def unzip(surr_name,sdir=None):

	### TODO: check that surr_name exists
	if ( sdir is None):
		unzip_me = download_path()
	else:
		unzip_me = sdir

	os.system('tar -xvzf '+unzip_me+surr_name+'.tar.gz')
	os.system('mv '+surr_name+ ' '+unzip_me)

	# returns location of unziped surrogate
	return unzip_me+surr_name+'/'
