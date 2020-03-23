#!/bin/bash

# @File Name : train.sh
# @Purpose :
# @Creation Date : 2020-03-21 15:12:49
# @Last Modified : 2020-03-23 09:15:41
# @Created By :  chenjiang
# @Modified By : chenjiang


# cache_dir="data/cache_check.tm12peppapig/"
# train_A_dir=data/vcc2018_training.speakers/VCC2TM1/
# train_B_dir=data/voice_material/peppapig/
# valid_A_dir=data/vcc2018_training.speakers/VCC2TM1/
# valid_B_dir=data/voice_material/peppapig/

cache_dir="data/cache_check.sf1_obama/"
train_A_dir="data/vcc2018_training.speakers/VCC2SF1/"
train_B_dir="data/voice_material/obama/"
valid_A_dir="data/vcc2018_training.speakers/VCC2SF1/"
valid_B_dir="data/voice_material/obama/"

python preprocess_training.py --train_A_dir ${train_A_dir} \
                              --train_B_dir ${train_B_dir} \
                              --cache_folder ${cache_dir} 

python train_cyclegan_vc2.py --logf0s_normalization   ${cache_dir}/logf0s_normalization.npz \
                             --mcep_normalization     ${cache_dir}/mcep_normalization.npz \
                             --coded_sps_A_norm       ${cache_dir}/coded_sps_A_norm.pickle \
                             --coded_sps_B_norm       ${cache_dir}/coded_sps_B_norm.pickle \
                             --model_checkpoint       ${cache_dir}/model_checkpoint/ \
                             --validation_A_dir       ${valid_A_dir} \
                             --output_A_dir           ${cache_dir}/converted_sound/VCC2TM1 \
                             --validation_B_dir       ${valid_B_dir} \
                             --output_B_dir           ${cache_dir}/converted_sound/peppapig/ \
                             --log_dir                ${cache_dir}/logs
                             # --resume_training_at     ${cache_dir}/model_checkpoint/_CycleGAN_CheckPoint \





