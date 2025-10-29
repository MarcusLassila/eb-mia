#!/bin/bash

if [[ ":${PYTHONPATH:-}:" != *":$PWD:"* ]]; then
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
fi
echo "PYTHONPATH=$PYTHONPATH"
pip install -e .
