"""gwsurrogate is the gravitational wave surrogate package


1) First load gwsurrogate 

    >>> import gwsurrogate as gw

2) Load a prebuilt surrogate model 

    >>> EOB = gw.SurrogateGW('EOBNRv2/SurrogateQ1to2/')

3) plot this waveform for some value of q_eval

    >>> EOB.plot(q_eval=1.2)

"""

__copyright__ = "Copyright (C) 2014 Scott Field"
__email__ = "sfield@umd.edu"
__status__ = "testing"
__author__ = "Scott Field"

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

from surrogate import *
