> **Fork notice** — this fork fine-tunes the action-conditioned iVideoGPT on [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) on a MacBook (Apple Silicon / MPS) and adds a held-out evaluation pipeline. Upstream: [thuml/iVideoGPT](https://github.com/thuml/iVideoGPT). Original README below the results.

# Bridge V2 fine-tune on Apple Silicon (this fork)

Fine-tuned the OXE-pretrained, action-free 64px model into an **action-conditioned (7-DoF) world model** on BridgeData V2 episodes, entirely on a MacBook via MPS. 3,000 steps, frozen tokenizer, 120 training / 20 held-out episodes.

## Results

Held-out rollouts: 40 segments from 20 unseen episodes, 2 context frames + 14 predicted frames, 64x64, seed 0 (`inference/eval_bridge.py`):

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ |
|---|---|---|---|---|
| **fine-tuned, action-conditioned** | **19.65 ± 2.87** | **0.741 ± 0.108** | **0.164 ± 0.069** | **445.5** |
| pretrained act-free (zero-shot) | 17.52 ± 2.23 | 0.684 ± 0.101 | 0.207 ± 0.068 | 548.2 |

Ground truth | fine-tuned | zero-shot baseline:

![rollout 0](assets/bridge_eval/val_eps_00000000-gt-vs-pred.gif)
![rollout 1](assets/bridge_eval/val_eps_00000001-gt-vs-pred.gif)
![rollout 2](assets/bridge_eval/val_eps_00000002-gt-vs-pred.gif)
![rollout 3](assets/bridge_eval/val_eps_00000003-gt-vs-pred.gif)

## Found: silent MPS numerical bug in the VQ decoder

On PyTorch 2.2 MPS, the pretrained `ctx_vqgan` **decoder** produces wrong pixels: pure tokenize→detokenize reconstruction drops to ~12–16 PSNR on MPS vs ~24–31 PSNR on CPU — silently, no errors. Token ids from `tokenize()` are **bit-identical** between CPU and MPS, and transformer logits match to 1e-6, so training on MPS is unaffected (loss is computed on token ids), but every decoded rollout, GIF, and pixel metric is corrupted. Workaround used here: generate on MPS, `detokenize()` on a CPU copy of the tokenizer (see `inference/eval_bridge.py`).

## What this fork adds

- `datasets/bridge_v2_extract.py` — extract Bridge V2 episodes from locally downloaded TFDS shards (works with partial downloads) to per-episode `.npz`
- `scripts/finetune/bridge-64-act-cond-mac.sh` — action-conditioned fine-tune config for MPS
- `inference/eval_bridge.py` — held-out evaluation: PSNR / SSIM / LPIPS / FVD + side-by-side GIFs, fine-tuned vs zero-shot baseline, with the CPU-decode workaround
- MPS fixes: device fallback in `inference/predict.py`, `attention_mask` for act-free generation on MPS, self-contained checkpoint export for the action-conditioned wrapper in `train_gpt.py`

## Reproduce

```bash
# 1. extract episodes (after downloading bridge_dataset TFDS shards into data/bridge_dataset)
python datasets/bridge_v2_extract.py --split train --output_path data/bridge_npz/train --max_num_episodes 120
python datasets/bridge_v2_extract.py --split val --output_path data/bridge_npz/val --max_num_episodes 20

# 2. fine-tune (Apple Silicon)
STEPS=3000 bash scripts/finetune/bridge-64-act-cond-mac.sh

# 3. evaluate
PYTHONPATH=.:inference python -u inference/eval_bridge.py --run_dir log_trm/<run-dir>
```

---

# 🌏 iVideoGPT: Interactive VideoGPTs are Scalable World Models (NeurIPS 2024)

[[Project Page]](https://thuml.github.io/iVideoGPT/) [[Paper]](https://arxiv.org/abs/2405.15223) [[Models]](https://huggingface.co/collections/thuml/ivideogpt-674c59cae32231024d82d6c5) [[Poster]](https://manchery.github.io/assets/pub/nips2024_ivideogpt/poster.pdf) [[Slides]](https://manchery.github.io/assets/pub/nips2024_ivideogpt/slides.pdf) [[Blog (In Chinese)]](https://mp.weixin.qq.com/s/D94aamdqtO9WLekr4BSCUw)

This repo provides official code and checkpoints for iVideoGPT, a generic and efficient world model architecture that has been pre-trained on millions of human and robotic manipulation trajectories. 

![architecture](assets/architecture.png)

## 🔥 News

- 🚩 **2025.09.18**: [RLVR-World](https://github.com/thuml/RLVR-World) has been accepted by NeurIPS 2025, congrats!
- 🚩 **2025.05.21**: We are excited to release a new work, [RLVR-World](https://github.com/thuml/RLVR-World), demonstrating that iVideoGPTs can be improved by reinforcement learning with verifiable rewards (RLVR)!
- 🚩 **2024.11.01**: NeurIPS 2024 camera-ready version is released on [arXiv](https://arxiv.org/abs/2405.15223v3).
- 🚩 **2024.09.26**: iVideoGPT has been accepted by NeurIPS 2024, congrats!
- 🚩 **2024.08.31**: Training code is released.
- 🚩 **2024.05.31**: Project website with video samples is released.
- 🚩 **2024.05.30**: Model pre-trained on Open X-Embodiment and inference code are released.
- 🚩 **2024.05.27**: Our paper is released on [arXiv](https://arxiv.org/abs/2405.15223v1).

## 🛠️ Installation

```bash
conda create -n ivideogpt python==3.9
conda activate ivideogpt
pip install -r requirements.txt
```

To evaluate the FVD metric, download the [pretrained I3D model](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1) into `pretrained_models/i3d/i3d_torchscript.pt`.

## 🤗 Models

At the moment we provide the following pre-trained models:

| Model | Resolution | Action-conditioned | Goal-conditioned | Tokenizer Size | Transformer Size |
| ---- | ---- | ---- | ---- | ---- | ---- |
| [ivideogpt-oxe-64-act-free](https://huggingface.co/thuml/ivideogpt-oxe-64-act-free) | 64x64 | No | No | 114M   |  138M    |
| [ivideogpt-oxe-64-act-free-medium](https://huggingface.co/thuml/ivideogpt-oxe-64-act-free-medium) | 64x64 | No | No |  114M   |  436M    |
| [ivideogpt-oxe-64-goal-cond](https://huggingface.co/thuml/ivideogpt-oxe-64-goal-cond) | 64x64 | No | Yes | 114M   |  138M    |
| [ivideogpt-oxe-256-act-free](https://huggingface.co/thuml/ivideogpt-oxe-256-act-free) | 256x256 | No | No | 310M   |  138M    |

If no network connection to Hugging Face, you can manually download from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/ef7d94c798504587a95e/).

**Notes**:

- Due to the heterogeneity of action spaces, we currently do not have an action-conditioned prediction model on OXE.
- Pre-trained models at 256x256 resolution may not perform best due to insufficient training, but can serve as a good starting point for downstream fine-tuning.

<details>
  <summary><b>More models on downstream tasks</b></summary>
  <br>
  
| Model | Resolution | Action-conditioned | Goal-conditioned | Tokenizer Size | Transformer Size |
| ---- | ---- | ---- | ---- | ---- | ---- |
| [ivideogpt-bair-64-act-free](https://huggingface.co/thuml/ivideogpt-bair-64-act-free) | 64x64 | No | No |  114M   |  138M    |
| [ivideogpt-bair-64-act-cond](https://huggingface.co/thuml/ivideogpt-bair-64-act-cond) | 64x64 | Yes | No | 114M   |  138M    |
| [ivideogpt-robonet-64-act-cond](https://huggingface.co/thuml/ivideogpt-robonet-64-act-cond) | 64x64 | Yes | No |  114M   |  138M    |
| [ivideogpt-vp2-robosuite-64-act-cond](https://huggingface.co/thuml/ivideogpt-vp2-robosuite-64-act-cond) | 64x64 | Yes | No |  114M   |  138M    |
| [ivideogpt-vp2-robodesk-64-act-cond](https://huggingface.co/thuml/ivideogpt-vp2-robodesk-64-act-cond) | 64x64 | Yes | No |  114M   |  138M    |

- We are sorry that the checkpoints for RoboNet at 256x256 resolution were deleted by mistake during a disk cleanup, we will retrain and release them as our computational resources become idle.
</details>

## 📦 Data Preparation

**Open X-Embodiment**: Download datasets from [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) and extract single episodes as `.npz` files:

```bash
python datasets/oxe_data_converter.py --dataset_name {dataset name, e.g. bridge} --input_path {path to downloaded OXE} --output_path {path to stored npz}
```

To replicate our pre-training on OXE, you need to extract all datasets listed under `OXE_SELECT` in `ivideogpt/data/dataset_mixes.py`.

See instructions at [`datasets`](/datasets) on preprocessing more datasets.

## 🚀 Inference Examples

For action-free video prediction on Open X-Embodiment, run:

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-64-act-free" --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data
```

See more examples at [`inference`](/inference).

## 🌟 Pre-training

To pre-train iVideoGPT, adjust the arguments in the command below as needed and run:

```bash
bash ./scripts/pretrain/ivideogpt-oxe-64-act-free.sh
```

See more scripts for [pre-trained models](#-models) at [`scripts/pretrain`](/scripts/pretrain).

## 🎇 Fine-tuning Video Prediction

### Finetuning Tokenizer

After preparing the [BAIR](/datasets#bair-robot-pushing) dataset, run the following:

```bash
accelerate launch train_tokenizer.py \
    --exp_name bair_tokenizer_ft --output_dir log_vqgan --seed 0 --mixed_precision bf16 \
    --model_type ctx_vqgan \
    --train_batch_size 16 --gradient_accumulation_steps 1 --disc_start 1000005 \
    --oxe_data_mixes_type bair --resolution 64 --dataloader_num_workers 16 \
    --rand_select --video_stepsize 1 --segment_horizon 16 --segment_length 8 --context_length 1 \
    --pretrained_model_name_or_path pretrained_models/ivideogpt-oxe-64-act-free/tokenizer \
    --max_train_steps 200005
```

### Finetuning Transformer

For action-conditioned video prediction, run the following:

```bash
accelerate launch train_gpt.py \
    --exp_name bair_llama_ft --output_dir log_trm --seed 0 --mixed_precision bf16 \
    --vqgan_type ctx_vqgan \
    --pretrained_model_name_or_path {log directory of finetuned tokenizer}/unwrapped_model \
    --config_name configs/llama/config.json --load_internal_llm --action_conditioned --action_dim 4 \
    --pretrained_transformer_path pretrained_models/ivideogpt-oxe-64-act-free/transformer \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --lr_scheduler_type cosine \
    --oxe_data_mixes_type bair --resolution 64 --dataloader_num_workers 16 \
    --video_stepsize 1 --segment_length 16 --context_length 1 \
    --use_eval_dataset --use_fvd --use_frame_metrics \
    --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd \
    --max_train_steps 100005
```

For action-free video prediction, remove `--load_internal_llm --action_conditioned`.

See more scripts at [`scripts/finetune`](/scripts/finetune).

### Evaluation

To evaluate the checkpoints only, run:

```bash
bash ./scripts/evaluation/bair-64-act-cond.sh
```

See more scripts for [released checkpoints](#-models) at [`scripts/evaluation`](/scripts/evaluation).

## 🤖 Visual Control

### Visual Model-based RL

Install the Metaworld version we used:

```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@83ac03ca3207c0060112bfc101393ca794ebf1bd
```

Modify paths in `mbrl/cfgs/mbpo_config.yaml` to your own paths (currently only support absolute paths).

Run model-based RL with iVideoGPT:

```bash
python mbrl/train_metaworld_mbpo.py task=plate_slide num_train_frames=100002 demo=true
```

### Visual Planning

See [`vp`](/vp) for detailed instructions.

## 🎥 Showcases

![showcase](assets/showcase.png)

## 📜 Citation

If you find this project useful, please cite our paper as:

```
@inproceedings{wu2024ivideogpt,
    title={iVideoGPT: Interactive VideoGPTs are Scalable World Models}, 
    author={Jialong Wu and Shaofeng Yin and Ningya Feng and Xu He and Dong Li and Jianye Hao and Mingsheng Long},
    booktitle={Advances in Neural Information Processing Systems},
    year={2024},
}
```

## 🤝 Contact

If you have any question, please contact wujialong0229@gmail.com.

## 💡 Acknowledgement

Our codebase is based on [huggingface/diffusers](https://github.com/huggingface/diffusers) and [facebookresearch/drqv2](https://github.com/facebookresearch/drqv2).
