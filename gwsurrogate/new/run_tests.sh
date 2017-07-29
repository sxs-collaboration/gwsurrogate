#!/usr/bin/env bash

for python_file in $(ls test*.py); do
  module=$(echo $python_file | rev | cut -c 4- | rev)
  echo $module
  python -m unittest $module
done
