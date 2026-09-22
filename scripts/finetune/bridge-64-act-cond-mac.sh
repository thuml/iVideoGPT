#!/bin/bash
# Fine-tune action-conditioned iVideoGPT transformer on BridgeData V2, Apple Silicon (MPS).
# Tokenizer stays frozen (OXE pretrained handles Bridge frames); only the transformer +
# new action head (action_dim=7) are trained. Run directly with python so Accelerate
# auto-selects MPS. Override STEPS / BS / GA via env.
set -e
cd "$(dirname "$0")/../.."

STEPS=${STEPS:-6}                    # smoke default; set STEPS=20000 for a real run
BS=${BS:-2}                          # per-device batch
GA=${GA:-4}                          # grad accumulation -> effective batch BS*GA
VAL=${VAL:-100000}                   # validation interval (high = only final-step eval)
GIF=${GIF:-100000}                   # gif-gen interval inside eval (high = skip slow gen)
EVAL_ITERS=${EVAL_ITERS:-2}          # cap eval batches

PYTHONPATH=. .venv/bin/python train_gpt.py \
    --exp_name bridge_act_cond_mac --output_dir log_trm --seed 0 --mixed_precision no \
    --vqgan_type ctx_vqgan \
    --pretrained_model_name_or_path pretrained_models/ivideogpt-oxe-64-act-free/tokenizer \
    --config_name configs/llama/config.json --load_internal_llm --action_conditioned --action_dim 7 \
    --pretrained_transformer_path pretrained_models/ivideogpt-oxe-64-act-free/transformer \
    --per_device_train_batch_size "$BS" --per_device_eval_batch_size 2 --gradient_accumulation_steps "$GA" \
    --learning_rate 1e-4 --lr_scheduler_type cosine \
    --oxe_data_mixes_type bridge --dataset_path data/bridge_npz --resolution 64 --dataloader_num_workers 0 \
    --video_stepsize 1 --segment_length 16 --context_length 2 \
    --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd \
    --max_train_steps "$STEPS" --log_steps 1 --checkpointing_steps 100000 \
    --validation_steps "$VAL" --skip_first_val --max_eval_iters "$EVAL_ITERS" --log_gif_interval "$GIF"
