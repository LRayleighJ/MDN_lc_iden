#!/bin/bash
#PBS -N TimeSeqKMT
#PBS -lselect=1:ncpus=20:mem=20gb
#PBS -o /home/zerui603/iden_lc_ML_ver2/log/
#PBS -e /home/zerui603/iden_lc_ML_ver2/log/
cd $PBS_O_WORKDIR

python /home/zerui603/iden_lc_ML_ver2/deal_with_KMTsimu/test_200fig.py
