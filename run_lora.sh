#!/bin/bash
output_dir="outputs" # output file
train_file="CKIP-Llama-2-7b/data/sample_train.jsonl" # training file
validation_file="CKIP-Llama-2-7b/data/sample_validation.jsonl" # validation file
export BELLE_DIR="./BELLE"
model_name_or_path="ckiplab/CKIP-Llama-2-7b" # or ckiplab/CKIP-Llama-2-7b-chat
cache_dir="hf_cache_dir"

mkdir -p ${output_dir}
mkdir -p ${cache_dir}

export CUDA_VISIBLE_DEVICES='0,1,2,3'
export WANDB_PROJECT="CKIP-Llama"
# export WANDB_ENTITY="WANDB_ENTITY"
# export WANDB_RESUME=allow
export PYTHONPATH="$BELLE_DIR/train"

# FT
torchrun --nproc_per_node 1 $BELLE_DIR/train/src/entry_point/sft_train.py \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --use_lora True \
    --use_int8_training \
    --lora_config $BELLE_DIR/train/configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 2 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 8e-6 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --fp16 True \
    --seed 1234 \
    --gradient_checkpointing True \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --use_flash_attention \
    --deepspeed $BELLE_DIR/train/configs/deepspeed_config_stage3.json\


    
