#!/bin/bash

# @File Name : train.sh
# @Purpose :
# @Creation Date : 2020-03-21 15:12:49
# @Last Modified : 2020-03-31 19:12:20
# @Created By :  chenjiang
# @Modified By : chenjiang


# cache_dir="data/cache_check.tm12peppapig/"
# train_A_dir=data/vcc2018_training.speakers/VCC2TM1/
# train_B_dir=data/voice_material/peppapig/
# valid_A_dir=data/vcc2018_training.speakers/VCC2TM1/
# valid_B_dir=data/voice_material/peppapig/

# cache_dir="data/cache_check.sf1_tm2.v3/"
# train_A_dir="data/vcc2018_training.speakers/VCC2SF1/"
# train_B_dir="data/vcc2018_training.speakers/VCC2TM2/"
# valid_A_dir="data/vcc2018_training.speakers/VCC2SF1/"
# valid_B_dir="data/vcc2018_training.speakers/VCC2TM2/"

# cache_dir="data/cache_check.sf3_tm1.sr22000/"
# train_A_dir="data/vcc2018_training.speakers/VCC2SF3/"
# train_B_dir="data/vcc2018_training.speakers/VCC2TM1/"
# valid_A_dir="data/vcc2018_training.speakers/VCC2SF3/"
# valid_B_dir="data/vcc2018_training.speakers/VCC2TM1/"

# cache_dir="data/cache_check.sf3_tm1.test"
# train_A_dir="data/vcc2018_training.speakers/VCC2SF3"
# train_B_dir="data/vcc2018_training.speakers/VCC2TM1/"
# valid_A_dir="data/vcc2018_training.speakers/VCC2SF3"
# valid_B_dir="data/vcc2018_training.speakers/VCC2TM1/"

cache_dir="data/cache_check.all_tm1/"
train_A_dir="data/vctk_vcc2018_peppapig/"
train_B_dir="data/vcc2018_training.speakers/VCC2TM1/"
valid_A_dir="data/vctk_vcc2018_peppapig/"
valid_B_dir="data/vcc2018_training.speakers/VCC2TM1/"

# cache_dir="data/cache_check.sf3_peppapigen/"
# train_A_dir="data/vcc2018_training.speakers/VCC2SF3/"
# train_B_dir="data/voice_material_new/PeppaPig_en"
# valid_A_dir="data/vcc2018_training.speakers/VCC2SF3/"
# valid_B_dir="data/voice_material_new/PeppaPig_en"

# cache_dir="data/cache_check.p270_p256/"
# train_A_dir="data/vctk_sub/p270/"
# train_B_dir="data/vctk_sub/p256/"
# valid_A_dir="data/vctk_sub/p270/"
# valid_B_dir="data/vctk_sub/p256/"

save_interval=100
if [[ "${cache_dir}" == *"all_"* ]];then
    save_interval=10
fi

# python preprocess_training.py --train_A_dir ${train_A_dir} \
#                               --train_B_dir ${train_B_dir} \
#                               --cache_folder ${cache_dir} 


python train_cyclegan_vc2.py --logf0s_normalization   ${cache_dir}/logf0s_normalization.npz \
                             --mcep_normalization     ${cache_dir}/mcep_normalization.npz \
                             --coded_sps_A_norm       ${cache_dir}/coded_sps_A_norm.pickle \
                             --coded_sps_B_norm       ${cache_dir}/coded_sps_B_norm.pickle \
                             --model_checkpoint       ${cache_dir}/model_checkpoint/ \
                             --validation_A_dir       ${valid_A_dir} \
                             --output_A_dir           ${cache_dir}/converted_sound/all/ \
                             --validation_B_dir       ${valid_B_dir} \
                             --output_B_dir           ${cache_dir}/converted_sound/VCC2TM1/ \
                             --log_dir                ${cache_dir}/logs \
                             --save_interval ${save_interval}
                             # --resume_training_at     ${cache_dir}/model_checkpoint/_CycleGAN_CheckPoint \





