#!/bin/bash
#PBS -N testIdiotNet
#PBS -lselect=1:ncpus=20:mem=128gb:ngpus=1
#PBS -o /home/zerui603/MDN_lc_iden/log/
#PBS -e /home/zerui603/MDN_lc_iden/log/
cd $PBS_O_WORKDIR

python /home/zerui603/MDN_lc_iden/unet/testdataset.py                              

