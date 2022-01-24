#!/bin/bash
#PBS -N gen_simudata_test
#PBS -lselect=1:ncpus=20:mem=256gb
#PBS -o /home/zerui603/MDN_lc/log/
#PBS -e /home/zerui603/MDN_lc/log/
cd $PBS_O_WORKDIR

for i in {0..7};do                       
for j in {0..7};do                       
    python /home/zerui603/MDN_lc/datagenerate/generate_test_bins.py $i $j                                   
done                                       
done                                       
