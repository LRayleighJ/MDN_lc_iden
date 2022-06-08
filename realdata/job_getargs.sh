#!/bin/bash
#PBS -N get_URLargs
#PBS -lselect=1:ncpus=20:mem=32gb
#PBS -o /home/zerui603/MDN_lc_iden/log/
#PBS -e /home/zerui603/MDN_lc_iden/log/
cd $PBS_O_WORKDIR

python /home/zerui603/MDN_lc_iden/realdata/download.py