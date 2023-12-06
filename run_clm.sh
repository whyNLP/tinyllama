export WANDB_PROJECT=Llama-test
echo "Start running..."
echo "Slurm job id: $SLURM_JOB_ID"

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000))
    echo $(($num%$max+$min))
}

MASTER_PORT=$(rand 50000 60000)

export WANDB_API_KEY=46989fba36f9c654c98a2435d2e33691fd3d3e4a

# export HF_ENDPOINT=https://hf-mirror.com

# improvement: ***** huge
export TINY_FLASH_ATTN=1
# improvement: ****  significant
export TINY_FUSED_RMSNORM=1
# improvement: *     minor
export TINY_FUSED_CROSSENTROPY=1
# improvement: *     minor
export TINY_FUSED_ROTARY=1
# improvement: **    a little bit
# export TINY_FUSED_SWIGLU=1

# benchmark
# tinyllama-tiny, RTX3090, 8 cards, wikitext103, 21 epoch
# 1, 2, 5 - bz32 - 20G? - 2h 45min
# 1, 2, 3, 4, 5 - bz32 - 15.9G - 2h 40min

# tinyllama, RTX3090, 8 cards, wikitext103, 30 epoch
# none - OOM
# 1 - bz1 - 11.8G - 101h 30min
# 1 - bz2 - 16.4G - 55h 30min
# 1 - bz4 - OOM
# 1, 2 - bz4 - 20.7G - 33h 20min
# 1, 2, 3, 4 - bz4 - 19.7G - 32h 40min
# 1, 2, 3, 4, 5 - bz4 - 16.3G - 32h 20min


deepspeed --master_port $MASTER_PORT run_clm.py \
    --deepspeed ds_config.json \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --auto_find_batch_size \
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
    --run_name llama-test \
    --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`

# llama-7b, layer 16, 19.9G
# + optim adamw_bnb_8bit 20.2G
# + optim adafactor 18.4G

# # tiny toy, for testing purpose only
# deepspeed --master_port $MASTER_PORT run_clm.py \
#     --deepspeed ds_config.json \
#     --model_type llama \
#     --config_name llama.json \
#     --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --auto_find_batch_size \
#     --block_size 2048 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.015 \
#     --learning_rate 9.5e-4 \
#     --weight_decay 6.6e-6 \
#     --optim adafactor \
#     --bf16 \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --num_train_epochs 21 \
#     --save_total_limit 3 \
#     --save_strategy epoch \
#     --evaluation_strategy epoch \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to none \
#     --run_name llama-tiny-test-acc-ref \
#     --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`
#     # --output_dir /home/wuhy/projects/algpt/ALGPT/outputs/llama-tiny-original-test