#!/bin/bash
#PBS -N testIdiotNet
#PBS -lselect=1:ncpus=20:mem=128gb:ngpus=2
#PBS -o /home/zerui603/MDN_lc_iden/log/
#PBS -e /home/zerui603/MDN_lc_iden/log/
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
CUDA_VISIBLE_DEVICES=3,4 python /home/zerui603/MDN_lc_iden/unet/testrealKMT.py                                       
conda deactivate