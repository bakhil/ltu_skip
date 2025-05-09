#!/bin/bash
#SBATCH -J alm
#SBATCH -o ./log/%j_alm.txt
#SBATCH --qos=regular
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --partition=a6
#SBATCH --ntasks-per-node=32
#SBATCH --mem=470000
#SBATCH --exclusive

JUMP_TO_LAYER=8
AUDIO_FILES_PATH_PREFIX="../../.."
DATA_JSON_PATH=../../../openaqa/data/closed_ended/all_closed_qa_filtered.json

export TRANSFORMERS_CACHE=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

output_dir="../exp/JUMP_TO_${JUMP_TO_LAYER}/stage3_all_close"
mkdir -p $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh


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