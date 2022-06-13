#!/bin/bash

# Define resolution
resolution='lowres'
gpu_id='-1'
dataset='malaria'

# Perform a smoke test to see all components and modules are installed well.
python3 -W ignore ./run.py $dataset $resolution $gpu_id

