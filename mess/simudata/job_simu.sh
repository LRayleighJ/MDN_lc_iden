#!/bin/bash
#PBS -N gen_simudata
#PBS -lselect=1:ncpus=20:mem=32gb
#PBS -o /home/zerui603/MDN_lc/log/
#PBS -e /home/zerui603/MDN_lc/log/
cd $PBS_O_WORKDIR

for i in {0..19};do
python /home/zerui603/MDN_lc/simudata/gen_simu.py $i
done
unset i