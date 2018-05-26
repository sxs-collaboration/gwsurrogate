# Welcome to GWSurrogate! #

GWSurrogate is an easy to use interface to gravitational wave surrogate models.

Surrogates provide a fast and accurate evaluation mechanism for gravitational
waveforms which would otherwise be found through solving differential 
equations. These equations must be solved in the ``building" phase, which 
was performed using other codes. For details see:

[1] Scott Field, Chad Galley, Jan Hesthaven, Jason Kaye, and Manuel Tiglio. 
`"Fast prediction and evaluation of gravitational waveforms using surrogate 
models". Phys. Rev. X 4, 031006 (2014). arXiv: gr-qc:1308.3565

If you find this package useful in your work, please cite reference [1] and, 
if available, the relevant paper describing the specific surrogate used.

gwsurrogate is available at https://pypi.python.org


# Installation #

gwsurrogate is a pure-Python module, thus installation is very easy. 

## From pip ##

The python package pip supports installing from PyPI (the Python Package 
Index). gwsurrogate can be installed to the standard location 
(e.g. /usr/local/lib/pythonX.X/dist-packages) with

```
>>> pip install gwsurrogate
```

## From source ##

Download and unpack gwsurrogate-X.X.tar.gz to any folder gws_folder of your 
choosing. The gwsurrogate module can be used immediately by adding

```
  import sys
  sys.path.append('absolute_path_to_gws_folder')
```

at the beginning of any script/notebook which uses gwsurrogate. 

Alternatively, if you are a bash or sh user, edit your .profile 
(or .bash_profile) file and add the line

```
  export PYTHONPATH=~absolute_path_to_gws_folder:$PYTHONPATH
```

For a "proper" installation into gws_folder run

```
>>> python setup.py install --prefix=absolute_path_to_gws_folder
```

and edit the PYTHONPATH environment variable as described above.


# Getting Started #

Please read the gwsurrogate docstring found in the __init__.py file
or from ipython with

```
>>> import gwsurrogate as gws
>>> gws?
```

Additional examples can be found in the accompanying Jupyter notebooks
located in the 'tutorial' folder. To open a notebook, for example
basics.ipynb, do

```
  >>> jupyter notebook basics.ipynb
```
from the directory 'notebooks'


# Where to find surrogates? #

Surrogates can be downloaded directly from gwsurrogate. 

For download instructions see the basics.ipynb Jupyter notebook. Also visit
the NR surrogate [database](https://www.black-holes.org/surrogates/).


# Tests #

If you have downloaded the entire project as a tar.gz file, from the 
top folder, do

```
>>> py.test
```

# NSF Support #

This package is based upon work supported by the National Science Foundation 
under PHY-1316424 and PHY-1208861.

Any opinions, findings, and conclusions or recommendations expressed in 
gwsurrogate are those of the authors and do not necessarily reflect the 
views of the National Science Foundation.
