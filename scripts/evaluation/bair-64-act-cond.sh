accelerate launch train_gpt.py \
    --exp_name bair-64-act-cond-eval --output_dir log_eval --seed 0 --mixed_precision bf16 \
    --vqgan_type ctx_vqgan --config_name configs/llama/config.json \
    --pretrained_model_name_or_path pretrained_models/ivideogpt-bair-64-act-cond/tokenizer \
    --pretrained_transformer_path pretrained_models/ivideogpt-bair-64-act-cond/transformer \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
    --oxe_data_mixes_type bair --resolution 64 --dataloader_num_workers 16 \
    --video_stepsize 1 --segment_length 16 --context_length 1 \
    --use_eval_dataset --use_fvd --use_frame_metrics \
    --eval_only --eval_generate_times 100 --max_generate_batchsize 80 --max_decode_batchsize 16 \
    --action_conditioned --action_dim 4 \
    >> bair-64-act-cond-eval.log 2>&1