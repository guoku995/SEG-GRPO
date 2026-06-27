# SEG-GRPO

- Authors: Guoku Jia, Yuan Liang, YiShi Chen, YanMei Meng, XiangNing Wu, Jonny Qin
- Institute: The University of Guangxi

This repository uses the GRPO algorithm to train Qwen-VL for referring image segmentation.

## Project Files

- `qwen_sam2_RL_wcot.py`: main training / running file.
- `qwen_sam2_eval.py`: evaluation file.

## Environment Setup

The current repository configuration is based on `GRPOSEG安装说明.txt`, `requirements.txt`, and the actual script imports.

Recommended environment:

- Python `3.12`
- CUDA `12.6`
- PyTorch `2.7.0`
- torchvision `0.22.0`
- torchaudio `2.7.0`
- transformers `4.57.0`

Create the environment:

```bash
conda create -n grpo python=3.12
conda activate grpo
```

Install PyTorch:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```

Install Transformers:

```bash
pip install transformers==4.57.0
```

Install FlashAttention:

```bash
pip install flash-attn --no-build-isolation
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Additional Dependencies

The code uses `SAM2ImagePredictor`, and the scripts also note that SAM2 requires extra dependencies such as:

```bash
pip install hydra-core iopath
```

If you do not already have the SAM2 package installed, install it according to the official SAM2 repository and make sure `from sam2.sam2_image_predictor import SAM2ImagePredictor` works in the current environment.

## Dataset Preparation

Supported referring segmentation datasets include:

- [refCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip)
- [refCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip)
- [refCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip)
- [COCO 2014 train images](http://images.cocodataset.org/zips/train2014.zip)

Place the dataset under the default `./dataset` directory with the following structure:

```text
dataset/
`-- refer_seg/
    |-- images/
    |   `-- mscoco/
    |       `-- images/
    |           `-- train2014/
    |-- refcoco/
    |-- refcoco+/
    `-- refcocog/
```

The training and evaluation scripts default to `--dataset_dir ./dataset`.

## Model Weights

This project depends on Qwen-VL base weights and SAM2 weights.

1. Qwen base model

The training script `qwen_sam2_RL_wcot.py` loads the base model from:

```text
checkpoints/4B
```

The evaluation script `qwen_sam2_eval.py` defaults to:

```text
checkpoints/qwen2.5_3B
```

So before running, prepare local model directories such as:

```text
checkpoints/
|-- 4B/
`-- qwen2.5_3B/
```

2. SAM2 model

By default, both scripts use:

```text
facebook/sam2-hiera-large
```

This can be downloaded automatically through Hugging Face, or you can provide a local path with `--segmentation_model_path`.

3. LoRA weights

Trained LoRA weights are available at:

- [Hugging Face: guoku/SEG-GRPO](https://huggingface.co/guoku/SEG-GRPO)
- [Mirror: hf-mirror](https://hf-mirror.com/guoku/SEG-GRPO)

The code also sets:

```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

So it will use the Hugging Face mirror by default.

## Training

Main training file:

```text
qwen_sam2_RL_wcot.py
```

Default training dataset:

```text
refcoco|unc|train
```

Run training:

```bash
python qwen_sam2_RL_wcot.py --dataset_dir ./dataset --train_dataset "refcoco|unc|train" --output_dir ./outputs
```

Common arguments:

- `--precision`: `fp32`, `bf16`, or `fp16` default is `bf16`
- `--dataset_dir`: dataset root, default `./dataset`
- `--train_dataset`: one of:
  - `refcoco|unc|train`
  - `refcoco+|unc|train`
  - `refcocog|umd|train`
  - `ReasonSeg|train`
- `--segmentation_model_path`: default `facebook/sam2-hiera-large`
- `--output_dir`: output directory

Training logs are written to:

```text
outputs/tb_logs
```

You can monitor them with:

```bash
tensorboard --logdir outputs/tb_logs
```

## Evaluation

Evaluation file:

```text
qwen_sam2_eval.py
```

Example evaluation command:

```bash
python qwen_sam2_eval.py --dataset_dir ./dataset --val_dataset "refcoco|unc|testA" --reasoning_model_path ./checkpoint-14500-0.3-2 --base_model_path ./checkpoints/qwen2.5_3B
```

Common arguments:

- `--precision`: `fp32`, `bf16`, or `fp16` default is `bf16`
- `--dataset_dir`: dataset root, default `./dataset`
- `--val_dataset`: one of:
  - `refcoco|unc|val`
  - `refcoco|unc|testA`
  - `refcoco|unc|testB`
  - `refcoco+|unc|val`
  - `refcoco+|unc|testA`
  - `refcoco+|unc|testB`
  - `refcocog|umd|val`
  - `refcocog|umd|test`
  - `ReasonSeg|test`
- `--segmentation_model_path`: default `facebook/sam2-hiera-large`
- `--reasoning_model_path`: LoRA or trained checkpoint path
- `--base_model_path`: base Qwen model path; if omitted, the script uses `checkpoints/qwen2.5_3B`

The evaluation script prints metrics such as `gIoU`, `cIoU`, inference time, token usage, and peak GPU memory.

## Notes

- The current code is written for GPU execution and uses `flash_attention_2`.
- `qwen_sam2_RL_wcot.py` expects the Qwen base model to exist locally at `checkpoints/4B`.
- `qwen_sam2_eval.py` expects the base model at `checkpoints/qwen2.5_3B` unless `--base_model_path` is explicitly provided.
- If model download is unstable, keep using the Hugging Face mirror or pre-download all required weights locally.
