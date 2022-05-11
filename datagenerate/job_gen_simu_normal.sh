#!/bin/bash
#PBS -N gen_simudata_test
#PBS -lselect=1:ncpus=20:mem=256gb
#PBS -o /home/zerui603/MDN_lc_iden/log/
#PBS -e /home/zerui603/MDN_lc_iden/log/
cd $PBS_O_WORKDIR

# for i in {0..7};do                       
for j in {0..1};do                       
    python /home/zerui603/MDN_lc_iden/datagenerate/generate_normal.py $j                                   
done                                       
# done 
