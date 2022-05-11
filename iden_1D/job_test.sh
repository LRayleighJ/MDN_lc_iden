#!/bin/bash
#PBS -N test_GRU_iden
#PBS -lselect=1:ncpus=8:mem=32gb
#PBS -o /home/zerui603/MDN_lc/log/
#PBS -e /home/zerui603/MDN_lc/log/
cd $PBS_O_WORKDIR


python /home/zerui603/MDN_lc/iden_1D/testnet.py
