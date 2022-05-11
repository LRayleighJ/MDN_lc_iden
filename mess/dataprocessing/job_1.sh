#!/bin/bash
#PBS -N TimeSeqKMT
#PBS -lselect=1:ncpus=20:mem=64gb
#PBS -o /home/zerui603/MDN_lc/log/
#PBS -e /home/zerui603/MDN_lc/log/
cd $PBS_O_WORKDIR

python /home/zerui603/MDN_lc/dataprocessing/deal1.py
