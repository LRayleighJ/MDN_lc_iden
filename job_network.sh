#!/bin/bash
#PBS -N resnet_kmt
#PBS -lselect=1:ncpus=20:mem=64gb:ngpus=2
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

# for i in {0..7};do                        
CUDA_VISIBLE_DEVICES=3,4 python /home/zerui603/MDN_lc_iden/iden_1D/testmixnet.py # $i                         
# done
# python /home/zerui603/MDN_lc_iden/iden_1D/mixdataset.py
# CUDA_VISIBLE_DEVICES=0,1 python /home/zerui603/MDN_lc_iden/unet/unet_kmt.py                         
                                                               
conda deactivate

# for i in {0..7};do                        
#     CUDA_VISIBLE_DEVICES=0,1,2,6 python /home/zerui603/MDN_lc_iden/iden_1D/drawhistline.py $i                         
# done
