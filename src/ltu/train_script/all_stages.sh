#!/bin/bash
#SBATCH -J alm
#SBATCH -o ./log/%j_alm.txt
#SBATCH --qos=regular
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --partition=a5
#SBATCH --ntasks-per-node=32
#SBATCH --mem=470000
#SBATCH --exclusive

JUMP_TO_LAYER=8

OUTPUT_DIR_PREFIX="../exp/JUMP_TO_${JUMP_TO_LAYER}/"$(date +"%Y-%m-%d-%H-%M-%S")
cp "$0" ${OUTPUT_DIR_PREFIX}/$(date +"%Y-%m-%d-%H-%M-%S").sh


export TRANSFORMERS_CACHE=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

############################### Stage 1 ################################################
AUDIO_FILES_PATH_PREFIX="../../.."
DATA_JSON_PATH=../../../openaqa/data/closed_ended/combine_cla_filtered.json

output_dir="${OUTPUT_DIR_PREFIX}/stage1_proj_cla"
mkdir -p $output_dir

NPROC_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=1234 ../finetune_jump.py \
    --base_model '../../../pretrained_mdls/vicuna_ltu/' \
    --data_path $DATA_JSON_PATH \
    --audio_files_path_prefix $AUDIO_FILES_PATH_PREFIX \
    --jump_to_layer $JUMP_TO_LAYER \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 32 \
    --num_epochs 2 \
    --learning_rate 1e-3 \
    --cutoff_len 108 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules ['dummy'] \
    --train_on_inputs \
    --group_by_length \
    --save_steps 500 \
    --trainable_params proj
########################################################################################

sleep 60

############################### Stage 2 ################################################
AUDIO_FILES_PATH_PREFIX="../../.."
DATA_JSON_PATH=../../../openaqa/data/closed_ended/combine_cla_filtered.json

output_dir="${OUTPUT_DIR_PREFIX}/stage2_all_cla"
mkdir -p $output_dir

NPROC_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=1234 ../finetune_jump.py \
    --base_model "../exp/JUMP_TO_${JUMP_TO_LAYER}/stage1_proj_cla/checkpoint-8000/pytorch_model.bin" \
    --data_path $DATA_JSON_PATH \
    --audio_files_path_prefix $AUDIO_FILES_PATH_PREFIX \
    --jump_to_layer $JUMP_TO_LAYER \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 16 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --cutoff_len 108 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --save_steps 2100 \
    --trainable_params all
########################################################################################

sleep 60

############################### Stage 3 ################################################
AUDIO_FILES_PATH_PREFIX="../../.."
DATA_JSON_PATH=../../../openaqa/data/closed_ended/all_closed_qa_filtered.json

output_dir="${OUTPUT_DIR_PREFIX}/stage3_all_close"
mkdir -p $output_dir

NPROC_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=1234 ../finetune_jump.py \
    --base_model "../exp/JUMP_TO_${JUMP_TO_LAYER}/stage2_all_cla/checkpoint-8400/pytorch_model.bin" \
    --data_path $DATA_JSON_PATH \
    --audio_files_path_prefix $AUDIO_FILES_PATH_PREFIX \
    --jump_to_layer $JUMP_TO_LAYER \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 108 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --save_steps 2200 \
    --trainable_params all
########################################################################################

sleep 60

################################ Stage 4 ################################################
AUDIO_FILES_PATH_PREFIX="../../.."
DATA_JSON_PATH=../../../openaqa/data/openaqa_5.6M_filtered.json

output_dir="${OUTPUT_DIR_PREFIX}/stage4_all_mix"
mkdir -p $output_dir

NPROC_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=1234 ../finetune_jump.py \
    --base_model "../exp/JUMP_TO_${JUMP_TO_LAYER}/stage3_all_close/checkpoint-6600/pytorch_model.bin" \
    --data_path $DATA_JSON_PATH \
    --audio_files_path_prefix $AUDIO_FILES_PATH_PREFIX \
    --jump_to_layer $JUMP_TO_LAYER \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 108 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --save_steps 2000 \
    --trainable_params all
########################################################################################