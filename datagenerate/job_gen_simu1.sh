#!/bin/bash
#PBS -N gen_simudata_test
#PBS -lselect=1:ncpus=5:mem=96gb
#PBS -o /home/zerui603/MDN_lc/log/
#PBS -e /home/zerui603/MDN_lc/log/
cd $PBS_O_WORKDIR

python /home/zerui603/MDN_lc/datagenerate/generate1.py