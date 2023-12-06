# Tiny Llama

This is a side project that follows all the acceleration tricks in [tinyllama](https://github.com/jzhang38/TinyLlama), with the minimal modification to the huggingface transformers code. This means that one can pretrain a tinyllama with [huggingface trainer](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) on RTX3090 / RTX4090 / A6000 / A100 without gradient checkpointing, and the training speed is comparable to the original tinyllama code.

I use the latest codes in [FlashAttention](https://github.com/Dao-AILab/flash-attention/). I'm not sure if the codes will be faster than the original tinyllama code. I also use deepspeed to accelerate the training.

## Benchmark

| Model     | GPU        | Batch Size Per GPU | GPU Memory | Speed (tokens/s) |
| --------- | ---------- | ------------------ | ---------- | ---------------- |
| tinyllama | 8*RTX3090  | 4                  | 16.3G      | 36k              |
| tinyllama | 4*A6000    | 8                  | 30G        | 35k              |
| tinyllama | 4*A6000    | 12                 | 39G        | 40k              |
| tinyllama | 8*A40      | 8                  | 30G        | 86k              |
| tinyllama | 8*A40      | 12                 | 39G        | 92k              |
| llama-7b  | 8*A40      | 1                  | 39.5G      | 4.7k             |
| llama-7b  | 8*A100-80G | 4                  | 60G        | 18k              |

That means you could train a chinchilla-optimal TinyLlama (1.1B param, 22B tokens) in 1 week with 4 A6000 or 8 RTX3090. I don't have access to A100, so I'd appreciate it if someone could test it.

## Installation

Change the cuda version if it is not compatible.

```sh
conda install pytorch==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install xformers --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

In python code, import the `tinyllama` module. The code will automatically replace the transformer modules with the optimized modules.

The flash attention 2 is supported by huggingface transformers. Just load the model with flash attention 2 enabled.

```diff
+ import tinyllama
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T")
- model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T")
+ model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T", use_flash_attention_2=True)

...
```

That's not over! You need to switch on the optimizing strategies by the environment variables. 

```diff
+ export TINY_FUSED_RMSNORM=1
+ export TINY_FUSED_CROSSENTROPY=1
+ export TINY_FUSED_ROTARY=1
+ export TINY_FUSED_SWIGLU=1

deepspeed --master_port 29500 run_clm.py \
    --deepspeed ds_config.json \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --auto_find_batch_size \
    --block_size 2048 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.02 \
    --learning_rate 5e-4 \
    --weight_decay 6.6e-6 \
    --bf16 \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 20 \
    --save_total_limit 3 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to none \
    --output_dir outputs/tiny-llama
```

Now the codes should be Blazingly Fast. The acceleration also applies to other llama models. I think it should be possible to pretrain a llama-7b on 8 A6000, but I haven't tested it yet.


