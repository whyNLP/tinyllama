# Tiny Llama

This is a side project that follows all the acceleration tricks in [tinyllama](https://github.com/jzhang38/TinyLlama), with the minimal modification to the huggingface transformers code. This means that one can pretrain a tinyllama with [huggingface trainer](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) on RTX3090 / RTX4090 / A6000 / A100 without gradient checkpointing, and the training speed is comparable to (or even higher than) the original tinyllama code.

I use the latest codes in [FlashAttention](https://github.com/Dao-AILab/flash-attention/). I also use [accelerate](https://github.com/huggingface/accelerate) to accelerate the training.

## News
- [24/08/31] Update to support the latest  `transformers` library. Fix some minor issues in fused swiglu implementation.
- [24/03/10] Change backend to `accelerate`. The `deepspeed` backend version is still available [here](https://github.com/whyNLP/tinyllama/tree/1694ad94c51a6bf18fc159332944dcee03610798).
- [23/12/06] Initial release.

## Benchmark

> [!NOTE]
> The results below are based on an older version of the code. The latest version may have slightly different results.

| Model     | GPU        | Batch Size Per GPU | GPU Memory | Speed (tokens/s) |
| --------- | ---------- | ------------------ | ---------- | ---------------- |
| tinyllama | 1*RTX3090  | 4                  | 22.3GiB    | 8.2k             |
| tinyllama | 8*RTX3090  | 3                  | 20.8GiB    | 32.7k            |
| tinyllama | 1*RTX4090  | 4                  | 23.1GiB    | 17k              |
| tinyllama | 1*A6000    | 8                  | 43.8GiB    | 12.2k            |
| tinyllama | 4*A6000    | 8                  | 44GiB      | 40.4k            |
| tinyllama | 8*A6000    | 8                  | 44.1GiB    | 73.3k            |
| tinyllama | 1*A100-80G | 16                 | 76.2GiB    | 26.6k            |
| tinyllama | 8*A100-80G | 16                 | 76.9GiB    | 204.5k           |
| tinyllama | 8*A100-80G | 16*8               | 77.0GiB    | 212.9k           |
| llama-7b  | 8*A100-80G | 1                  | 78.8GiB    | 22.4k            |

<details>
<summary>Deepspeed Results</summary>

| Model     | GPU        | Batch Size Per GPU | GPU Memory | Speed (tokens/s) |
| --------- | ---------- | ------------------ | ---------- | ---------------- |
| tinyllama | 8*RTX3090  | 4                  | 16.3G      | 36k              |
| tinyllama | 4*A6000    | 8                  | 30G        | 35k              |
| tinyllama | 4*A6000    | 12                 | 39G        | 40k              |
| tinyllama | 8*A40      | 8                  | 30G        | 86k              |
| tinyllama | 8*A40      | 12                 | 39G        | 92k              |
| llama-7b  | 8*A40      | 1                  | 39.5G      | 4.7k             |
| llama-7b  | 8*A100-80G | 4                  | 60G        | 18k              |
</details>

where 16*8 means a batch size of 16 per GPU and gradient accumulation steps of 8. It achieves a throughput of 26.6k tokens per second per A100-80G GPU.

That means you could train a chinchilla-optimal TinyLlama (1.1B param, 22B tokens) in 1 week with 4 A6000 or 8 RTX3090, or 28.7 hours with 8 A100-80G.

## Installation

Use a proper python version, install pytorch, xformers, and other dependencies.

```sh
# for example, use python 3.10
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install xformers -c xformers
pip install -r requirements.txt
```

Change the cuda version if it is not compatible.

<details>
<summary>Development Environment</summary>

The current implementation is developed using python 3.12.4.

```sh
# install pytorch (2.5.0 nightly)
conda install pytorch pytorch-cuda=12.1 -c pytorch-nightly -c nvidia

# install xFormers
# (Optional) Makes the build much faster
pip install ninja
# Set TORCH_CUDA_ARCH_LIST if running and building on different GPU types
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
# (this can take dozens of minutes)

# install other dependencies like transformers, accelerate, etc.
pip install -r requirements.txt
```

> [!NOTE]
> I am using pytorch nightly build since I want to use FlexAttention. Feel free to use the stable version of [PyTorch](https://pytorch.org/get-started/locally/). If so, you can install [xFormers](https://github.com/facebookresearch/xformers) by `conda install xformers -c xformers`.

</details>

## Usage

### Option 1: Use our scripts

We have already provide you with the scripts to do training and inference. Simply run the following command.

```sh
bash run_clm.sh
```

See `run_clm.sh` for more details.

### Option 2: Modify your own code

In python code, import the `tinyllama` module. The code will automatically replace the transformer modules with the optimized modules.

The flash attention 2 is supported by huggingface transformers. Just load the model with flash attention 2 enabled.

```diff
+ import tinyllama
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T")
- model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T")
+ model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T", attn_implementation="flash_attention_2")

...
```

That's not over! You need to switch on the optimizing strategies by the environment variables. 

```diff
+ export TINY_FUSED_RMSNORM=1
+ export TINY_FUSED_CROSSENTROPY=1
+ export TINY_FUSED_ROTARY=1
+ export TINY_FUSED_SWIGLU=1

accelerate launch run_clm.py \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T \
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
    --torch_dtype bfloat16 \
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

Now the codes should be Blazingly Fast. The acceleration also applies to other llama models.


## Related Projects
- [tinyllama-zh](https://github.com/whyNLP/tinyllama-zh): My demo project that uses this repo to pretrain a Chinese TinyLlama.
- [LCKV](https://github.com/whyNLP/LCKV): My project that uses this repo to do pretraining on a variant of Llama model.
