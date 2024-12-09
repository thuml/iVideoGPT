# Pre-training tokenizer using four A100-40GB GPUs
# Arguments:
#   oxe_data_mixes_type: 'select_sthsth' for OXE+SSv2, 'select' for OXE only
#   dataset_path: path to preprocessed OXE dataset
#   sthsth_root_path: path to preprocessed SSv2 dataset

accelerate launch train_tokenizer.py \
    --exp_name oxe-64-act-free-tokenizer --output_dir log_vqgan --seed 0 --mixed_precision bf16 \
    --model_type ctx_vqgan \
    --learning_rate 5e-4 --discr_learning_rate 5e-4 \
    --train_batch_size 16 --gradient_accumulation_steps 1 --disc_start 1000005 \
    --oxe_data_mixes_type select_sthsth --resolution 64 --dataloader_num_workers 16 \
    --rand_select --video_stepsize 1 --segment_horizon 16 --segment_length 8 --context_length 2 \
    --dataset_path {path to preprocessed_OXE} \
    --sthsth_root_path {path to preprocessed_SSv2}


# Pre-training transformer using four A100-40GB GPUs
# Argments:
#   pretrained_model_name_or_path: path to the pre-trained tokenizer
accelerate launch train_gpt.py \
    --exp_name oxe-64-act-free-transformer --output_dir log_trm --seed 0 --mixed_precision bf16 \
    --vqgan_type ctx_vqgan \
    --pretrained_model_name_or_path {log directory of finetuned tokenizer}/unwrapped_model \
    --config_name configs/llama/config.json \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --lr_scheduler_type cosine \
    --oxe_data_mixes_type select --resolution 64 --dataloader_num_workers 16 \
    --dataset_path {path to preprocessed_OXE} \
    --video_stepsize 1 --segment_length 16 --context_length 2 \
    --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd