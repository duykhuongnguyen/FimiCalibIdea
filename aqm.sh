#!/bin/bash

/Users/anhduy0911/miniforge3/envs/torch_m1/bin/python /Users/anhduy0911/Projects/Lab/Fimi/Calibration/ModelIdeas/forgan_multistep.py -ds aqm -t lstm -steps 5000 -n 3 -c 7 -p 3 -rg 64 -rd 256 -d_iter 6 -hbin 50 -type train -best=True -metric rmse -raw True -mul True