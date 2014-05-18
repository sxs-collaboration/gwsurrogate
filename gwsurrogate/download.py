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

### dictionary of all known surrogates ###
### TODO: better way? should include information about this. What do package managers do? 
_surrogate_world = {'EOB1':'https://www.dropbox.com/s/4dpiw0zzh1wddmq/EOBNRv2.tar.gz',\
                   'spec1':'https://www.dropbox.com/s/kzqavkjfjq8wphr/spec_test.tar.gz'}

def list():

	for surr_id, surr_url in _surrogate_world.iteritems():
		print surr_id+' is located at '+surr_url

def get(surr_name,sdir='surrogate_downloads/'):

	###TODO: check that surr_name is valid. anything else to check? 
	os.system('wget --directory-prefix='+sdir+' '+_surrogate_world[surr_name])

def unzip(surr_name,sdir='surrogate_downloads/'):

	### TODO: check that surr_name exists
	os.system('tar -xvzf '+sdir+surr_name+'.tar.gz')
	os.system('mv '+surr_name+ ' '+sdir)

	# returns location of unziped surrogate
	return os.getcwd()+'/'+sdir+surr_name+'/'
