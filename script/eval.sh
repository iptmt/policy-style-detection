#!/bin/bash

echo "dataset: $1"
echo "file: $2"
echo "================================="

python acc.py $1 $2
python sim.py $1 $2
python ppl.py $1 $2