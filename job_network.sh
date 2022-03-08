#!/bin/bash
#PBS -N net_iden_lc
#PBS -lselect=1:ncpus=8:mem=16gb:ngpus=1
#PBS -o /home/zerui603/MDN_lc/log/
#PBS -e /home/zerui603/MDN_lc/log/
cd $PBS_O_WORKDIR

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate PytorchCd11
for i in {0..7};do                       
    CUDA_VISIBLE_DEVICES=0 python /home/zerui603/MDN_lc_iden/iden_1D/test_realKMT.py $i                               
done 
conda deactivate
