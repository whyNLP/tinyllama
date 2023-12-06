#!/bin/bash
# This script is used to run CLM training, evaluation, and prediction using TinyLlama-1.1B model.

echo "Start running..."

# improvement: ***** huge
export TINY_FLASH_ATTN=1
# improvement: ****  significant
export TINY_FUSED_RMSNORM=1
# improvement: *     minor
export TINY_FUSED_CROSSENTROPY=1
# improvement: *     minor
export TINY_FUSED_ROTARY=1
# improvement: **    a little bit
export TINY_FUSED_SWIGLU=1


deepspeed --master_port 29500 run_clm.py \
    --deepspeed ds_config.json \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --auto_find_batch_size \
    --gradient_accumulation_steps 1 \
    --block_size 2048 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.015 \
    --learning_rate 9.5e-4 \
    --weight_decay 6.6e-6 \
    --bf16 \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 30 \
    --save_total_limit 3 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to none \
    --output_dir outputs/tiny-llama
