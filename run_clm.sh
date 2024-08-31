#!/bin/bash
# This script is used to run CLM training, evaluation, and prediction using TinyLlama-1.1B model.

echo "Start running..."

# improvement: ***   good        (speed 0.0%, memory 7.0%)
export TINY_FLASH_ATTN=1
# improvement: ***** huge        (speed 8.7%, memory 16.2%)
export TINY_FUSED_RMSNORM=1
# improvement: **    normal      (speed 0.0%, memory 4.6%)
export TINY_FUSED_CROSSENTROPY=1
# improvement: *     minor       (speed 1.9%, memory 0.5%)
export TINY_FUSED_ROTARY=1
# improvement: ****  significant (speed 1.6%, memory 16.2%)
export TINY_FUSED_SWIGLU=1


accelerate launch run_clm.py \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T \
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
    --torch_dtype bfloat16 \
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
