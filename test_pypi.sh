#!/bin/bash

# test pipy installation works
# run this script in an env without an existing gwsurrogate installation 
pip show gwsurrogate
pip show gwtools
echo "if gwsurrogate or gwtools is installed you should stop this test"
sleep 3

cd ../../ # move into non-gwsurrogate directory

pip install gwsurrogate

python -c "import gwsurrogate as gws"

pip uninstall gwsurrogate
pip uninstall gwtools
