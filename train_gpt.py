import copy
import os
import warnings

current_path = os.getcwd()
print("current_path is: ", current_path)

import numpy as np
import torch
from tqdm import tqdm, trange
import time
import sys

import argparse
import json
import logging
import math
import os
from pathlib import Path
import imageio

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from tqdm.auto import tqdm

from safetensors.torch import load_file
import transformers
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from ivideogpt.vq_model import CompressiveVQModel
from ivideogpt.transformer import HeadModelWithAction
from ivideogpt.utils.video_metric import Evaluator, FeatureStats
from ivideogpt.data import *
from peft import LoraConfig, TaskType, get_peft_model

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.39.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())


def get_dataloaders(args):
    # DataLoaders creation:
    if args.strong_aug:
        augmentation_args = {
            'brightness': [0.6, 1.4],
            'contrast': [0.6, 1.4],
            'saturation': [0.6, 1.4],
            'hue': [-0.5, 0.5],
            'random_resized_crop_scale': (0.6, 1.0),
            'random_resized_crop_ratio': (0.75, 1.3333),
            'no_aug': args.no_aug,
        }
    else:
        augmentation_args = {
            'brightness': [0.9, 1.1],
            'contrast': [0.9, 1.1],
            'saturation': [0.9, 1.1],
            'hue': [-0.05, 0.05],
            'random_resized_crop_scale': (0.8, 1.0),
            'random_resized_crop_ratio': (0.9, 1.1),
            'no_aug': args.no_aug,
        }
    segment_args = {
        'random_selection': False,
        'random_shuffle': False,
        'goal_conditioned': args.goal_conditioned,
        'segment_length': args.segment_length,
        'context_length': args.context_length,
        'stepsize': args.video_stepsize,
        'segment_horizon': None,
    }
    train_dataloader = SimpleRoboticDataLoaderv2(
        parent_dir=args.dataset_path,
        datasets=DATASET_NAMED_MIXES[args.oxe_data_mixes_type],
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        train=True,
        maxsize=args.dataset_size,
        image_size=args.resolution,
        sthsth_root_path=args.sthsth_root_path,
        **augmentation_args,
        **segment_args,
        load_action=args.action_conditioned,
    )
    if args.use_eval_dataset:
        assert len(DATASET_NAMED_MIXES[args.oxe_data_mixes_type]) == 1
        eval_dataloader = EvalDataLoader(
            dataset_name=DATASET_NAMED_MIXES[args.oxe_data_mixes_type][0][0],
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.dataloader_num_workers,
            image_size=args.resolution,
            segment_length=args.segment_length,
            load_action=args.action_conditioned,
        )
    else:
        eval_dataloader = SimpleRoboticDataLoaderv2(
            parent_dir=args.dataset_path,
            datasets=DATASET_NAMED_MIXES[args.oxe_data_mixes_type],
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.dataloader_num_workers,
            train=False,
            image_size=args.resolution,
            sthsth_root_path=args.sthsth_root_path,
            **augmentation_args,
            **segment_args,
            load_action=args.action_conditioned,
        )
    return train_dataloader, eval_dataloader


def get_tokenizer(args):
    if args.vqgan_type == 'vqgan':
        raise NotImplementedError
        vq_model = VQModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder=None, revision=None, variant=None, use_safetensor=True
        ).eval()
        vocab_size = vq_model.num_vq_embeddings
    elif args.vqgan_type == 'ctx_vqgan':
        vq_model = CompressiveVQModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder=None, revision=None, variant=None, use_safetensor=True,
            low_cpu_mem_usage=False, device_map=None,
        ).eval()
        if args.context_length != vq_model.context_length:
            print(
                f"[Warning] pretrained context length of vq_model mismatch, change from {vq_model.context_length} to {args.context_length}")
            vq_model.set_context_length(args.context_length)
        vocab_size = vq_model.num_vq_embeddings + vq_model.num_dyn_embeddings
        if args.special_token:
            vocab_size += 2
    else:
        raise NotImplementedError
    return vq_model, vocab_size


def generate_multiple_times(
    gen_times,
    accelerator,
    model,
    gen_input,
    actions,
    gen_kwargs,
    max_batch_size=None,
    verbose=False,
    reward_prediction=False,
):
    max_batch_size = max_batch_size or gen_input.shape[0]
    assert max_batch_size % gen_input.shape[0] == 0
    repeat_times = max_batch_size // gen_input.shape[0]
    assert gen_times % (max_batch_size // gen_input.shape[0]) == 0
    repeat_iters = gen_times // (max_batch_size // gen_input.shape[0])
    results = []
    rewards = []
    for i in trange(repeat_iters, disable=not verbose):
        if reward_prediction:
            generated_tokens, reward = accelerator.unwrap_model(model).generate(
                gen_input.repeat(repeat_times, 1),
                **gen_kwargs,
                **({'action': actions.repeat(repeat_times, 1, 1)} if actions is not None else {}),
                pad_token_id=50256,  # this is meaningless but supressing warning
            )
            results.append(generated_tokens)
            rewards.append(reward)
        else:
            generated_tokens = accelerator.unwrap_model(model).generate(
                gen_input.repeat(repeat_times, 1),
                **gen_kwargs,
                **({'action': actions.repeat(repeat_times, 1, 1)} if actions is not None else {}),
                pad_token_id=50256,  # this is meaningless but supressing warning
            )
            results.append(generated_tokens)
    if reward_prediction:
        return torch.cat(results, dim=0), torch.cat(rewards, dim=0)
    results = torch.cat(results, dim=0)  # [t*B, ...] where t means number of generation times
    return results


def batch_forward(batch_size, input, forward, verbose=False):
    return torch.cat([forward(input[i: i + batch_size]) for i in trange(0, input.shape[0], batch_size, disable=not verbose)], dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--config_name", type=str, default="configs/llama/config.json",
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument('--llama_attn_drop', default=None, type=float)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=1000000,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="constant_with_warmup",
                        help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=5000,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default="trm-output", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--vqgan_type", type=str, default="vqgan",
                        choices=['vqgan', 'ctx_vqgan'], help="VQGAN model type to use.")
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True)
    parser.add_argument('--pretrained_transformer_path', type=str, default=None)
    parser.add_argument('--load_internal_llm', default=False, action='store_true')
    parser.add_argument("--trust_remote_code", type=bool, default=False,
                        help=(
                            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                            "execute code present on the Hub on your local machine."
                        ),
                        )
    parser.add_argument("--checkpointing_steps", type=int, default=5000,
                        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--with_tracking", type=bool, default=True,
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
                            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
                            "Only applicable when `--with_tracking` is passed."
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
    )
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--lora', default=False, action='store_true')
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=32, type=float)
    parser.add_argument('--lora_dropout', default=0.0, type=float)
    parser.add_argument('--gradient_checkpointing', default=False, action='store_true')
    parser.add_argument("--max_grad_norm", default=None, type=float, help="Max gradient norm.")

    parser.add_argument('--reward_prediction', default=False, action='store_true')
    parser.add_argument('--start_completed_steps', default=None, type=int)
    parser.add_argument('--action_recon', default=None, type=float)

    # datasets
    parser.add_argument("--segment_length", type=int, default=2,
                        help="The length of the segmented trajectories to use for the training.")
    parser.add_argument("--context_length", type=int, default=1)
    parser.add_argument('--video_stepsize', default=1, type=int)
    parser.add_argument('--dataset_path', default='/data2/frame_datasets',
                        type=str, help='Path to the tensorflow datasets')
    parser.add_argument('--dataset_size', default=None, type=int)
    parser.add_argument('--sthsth_root_path',
                        default='/data/something-something-v2/20bn-something-something-v2-frames-64', type=str)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help=(
                            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
                        ),
                        )
    parser.add_argument('--strong_aug', default=False, action='store_true')
    parser.add_argument('--no_aug', default=False, action='store_true')
    parser.add_argument('--oxe_data_mixes_type', default='select', type=str)

    parser.add_argument("--log_steps", type=int, default=100, help=("Print logs every X steps."))
    parser.add_argument("--validation_steps", type=int, default=5000)
    parser.add_argument('--skip_first_val', default=False, action='store_true')
    parser.add_argument('--latest_checkpoint_only', default=False, action='store_true')
    parser.add_argument('--special_token', default=True, action='store_true')
    parser.add_argument('--action_conditioned', default=False, action='store_true')
    parser.add_argument('--action_dim', default=4, type=int, help='action dimension for the task')
    parser.add_argument('--embed_no_wd', default=False, action='store_true')
    parser.add_argument('--goal_conditioned', default=False, action='store_true')

    # evaluation
    parser.add_argument('--max_eval_iters', default=100, type=int)
    parser.add_argument('--use_eval_dataset', default=False, action='store_true')
    parser.add_argument('--i3d_path', default='pretrained_models/i3d/i3d_torchscript.pt', type=str, help='path to the i3d model')
    parser.add_argument('--use_frame_metrics', default=False, action='store_true')
    parser.add_argument('--use_fvd', default=False, action='store_true')
    parser.add_argument('--eval_generate_times', default=1, type=int, help='for eval, fvd')
    parser.add_argument('--max_generate_batchsize', default=None, type=int)
    parser.add_argument('--max_decode_batchsize', default=None, type=int)
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--log_gif_interval', default=10, type=int)

    args = parser.parse_args()
    args.model_type = args.config_name.split('/')[1]
    if args.model_type not in ['gpt2', 'llama']:
        assert False, f"model_type {args.model_type} is not supported."

    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = args.per_device_train_batch_size

    assert not (args.action_conditioned and not args.special_token), \
        "Action conditioned model must have special token enabled."

    return args


@torch.no_grad
def evaluate(args, accelerator, tokenizer, model, eval_dataloader, evaluator, completed_steps):
    model.eval()
    losses = []
    mse_values, psnr_values, ssim_values, lpips_values, = [], [], [], []
    real_feats, gen_feats = FeatureStats(capture_mean_cov=True), FeatureStats(capture_mean_cov=True)
    eval_iters = min(len(eval_dataloader), args.max_eval_iters)
    bar = tqdm(range(eval_iters), desc="validation", disable=not accelerator.is_local_main_process)

    for i, batch in enumerate(eval_dataloader):
        if i == args.max_eval_iters:
            break
        if args.action_conditioned:
            pixel_values, actions = batch
            actions = actions.to(accelerator.device, non_blocking=True)
            pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
        else:
            pixel_values = batch.to(accelerator.device, non_blocking=True)
        batch_size = pixel_values.shape[0]

        if args.use_fvd:
            detector_kwargs = dict(rescale=True, resize=True, return_features=True)
            images = pixel_values.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, c, t, h, w]
            if args.max_decode_batchsize is not None and images.shape[0] > args.max_decode_batchsize:
                features = batch_forward(
                    args.max_decode_batchsize,
                    images * 255.,
                    lambda x: accelerator.unwrap_model(evaluator).i3d_model(x, **detector_kwargs)
                )
            else:
                features = accelerator.unwrap_model(evaluator).i3d_model(images * 255., **detector_kwargs)
            gathered_features = accelerator.gather(features)
            if accelerator.is_main_process:
                real_feats.append_torch(gathered_features)

        tokens, labels = accelerator.unwrap_model(tokenizer).tokenize(pixel_values,
                                                                      args.context_length,
                                                                      #   special_token=args.special_token
                                                                      )
        model_input = {'input_ids': tokens, 'labels': labels}
        if args.action_conditioned:
            model_input['action'] = actions

        if args.reward_prediction:
            if accelerator.num_processes > 1:
                outputs, rewards = model.module(**model_input)
            else:
                outputs, rewards = model(**model_input)
        else:
            if accelerator.num_processes > 1:
                outputs = model.module(**model_input)
            else:
                outputs = model(**model_input)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

        # predict next frames
        if (i % args.log_gif_interval == 0 and accelerator.is_main_process) or args.use_frame_metrics or args.use_fvd:
            if args.special_token:
                gen_input = tokens[:, :args.context_length * (256 + 1)]  # TODO: magic number
                # gen_input = tokens[:, :2 * (256 + 1)]  # TODO: magic number
                max_new_tokens = (1 + 16) * (args.segment_length - args.context_length) - 1
            else:
                gen_input = tokens[:, :args.context_length * 256]
                max_new_tokens = 16 * (args.segment_length - args.context_length)
            # generated_tokens = accelerator.unwrap_model(model).generate(
            #     gen_input,
            #     do_sample=True,
            #     temperature=1.0,
            #     top_k=100,
            #     max_new_tokens=max_new_tokens,
            #     **({'action': actions} if args.action_conditioned else {})
            # )
            if args.reward_prediction:
                generated_tokens, rewards = generate_multiple_times(
                    args.eval_generate_times,
                    accelerator, model, gen_input, actions if args.action_conditioned else None,
                    gen_kwargs={
                        'do_sample': True,
                        'temperature': 1.0,
                        'top_k': 100,
                        'max_new_tokens': max_new_tokens,
                    },
                    max_batch_size=args.max_generate_batchsize,
                    verbose=False,
                    # verbose=True,
                    reward_prediction=True,
                )
            else:
                generated_tokens = generate_multiple_times(
                    args.eval_generate_times,
                    accelerator, model, gen_input, actions if args.action_conditioned else None,
                    gen_kwargs={
                        'do_sample': True,
                        'temperature': 1.0,
                        'top_k': 100,
                        'max_new_tokens': max_new_tokens,
                    },
                    max_batch_size=args.max_generate_batchsize,
                    verbose=False,
                    # verbose=True,
                    reward_prediction=False,
                )
            if args.max_decode_batchsize is not None and generated_tokens.shape[0] > args.max_decode_batchsize:
                recon_output = batch_forward(
                    args.max_decode_batchsize,
                    generated_tokens,
                    lambda x: accelerator.unwrap_model(tokenizer).detokenize(
                        x, args.context_length,
                        # special_token=args.special_token
                    )
                )
            else:
                recon_output = accelerator.unwrap_model(tokenizer).detokenize(
                    generated_tokens, args.context_length,
                    # special_token=args.special_token
                )  # generated_tokens will include gen_input
            recon_output = recon_output.clamp(0.0, 1.0)

        # save predicted video
        if i % args.log_gif_interval == 0 and accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, "images", f"val-samples-{completed_steps}")
            os.makedirs(save_path, exist_ok=True)
            gt_frames = [(pixel_values[0, i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                         for i in range(pixel_values.shape[1])]
            recon_frames = [(recon_output[0, i].permute(1, 2, 0).detach().cpu().numpy() *
                             255).astype(np.uint8) for i in range(recon_output.shape[1])]
            frames = [np.concatenate([gt_frames[i], recon_frames[i], np.abs(gt_frames[i].astype(
                float) - recon_frames[i].astype(float)).astype(np.uint8)]) for i in range(len(gt_frames))]
            imageio.mimsave(f"{save_path}/val-samples-{completed_steps}-{i}.gif", frames, fps=4, loop=0)
            if not args.use_frame_metrics:
                assert pixel_values.shape[0] == recon_output.shape[0]
                mse_values.append(torch.mean((pixel_values - recon_output) ** 2).repeat(batch_size))

        if args.use_fvd:
            detector_kwargs = dict(rescale=True, resize=True, return_features=True)
            images = recon_output.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, c, t, h, w]
            if args.max_decode_batchsize is not None and images.shape[0] > args.max_decode_batchsize:
                features = batch_forward(
                    args.max_decode_batchsize,
                    images * 255.,
                    lambda x: accelerator.unwrap_model(evaluator).i3d_model(x, **detector_kwargs)
                )
            else:
                features = accelerator.unwrap_model(evaluator).i3d_model(images * 255., **detector_kwargs)
            gathered_features = accelerator.gather(features)
            if accelerator.is_main_process:
                gen_feats.append_torch(gathered_features)
                print("current fvd", accelerator.unwrap_model(evaluator).compute_fvd(real_feats, gen_feats))

        if args.use_frame_metrics:
            mse_value, psnr_value, ssim_value, lpips_value = evaluator(pixel_values.clamp(
                0.0, 1.0), recon_output)  # pixel_values can be 1.0000001192092896 numerically

            mse_values.append(accelerator.gather(mse_value.repeat(batch_size)))
            psnr_values.append(accelerator.gather(psnr_value.repeat(batch_size)))
            ssim_values.append(accelerator.gather(ssim_value.repeat(batch_size)))
            lpips_values.append(accelerator.gather(lpips_value.repeat(batch_size)))

        bar.update(1)

    if accelerator.is_main_process:
        try:
            eval_loss = torch.cat(losses, 0).mean().item()
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        eval_logs = {
            'eval/eval_loss': eval_loss,
            'eval/perplexity': perplexity,
            'eval/mse': torch.cat(mse_values, 0).mean().item(),
        }

        if args.use_fvd:
            fvd = accelerator.unwrap_model(evaluator).compute_fvd(real_feats, gen_feats)
            eval_logs.update({'eval/fvd': fvd})

        if args.use_frame_metrics:
            eval_logs.update({
                'eval/psnr': torch.cat(psnr_values, 0).mean().item(),
                'eval/ssim': torch.cat(ssim_values, 0).mean().item(),
                'eval/lpips': torch.cat(lpips_values, 0).mean().item(),
            })
        accelerator.log(eval_logs, step=completed_steps)

    model.train()

    if accelerator.is_main_process:
        return eval_logs
    else:
        return None


def plot_gif(x, postfix=''):
    # [B, T, C, H, W]
    frames = [(x[0, i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8) for i in range(x.shape[1])]
    imageio.mimsave(f"tmp{postfix}.gif", frames, fps=4, loop=0)


def start_train():
    args = parse_args()
    args.output_dir = os.path.join(args.output_dir, time.strftime("%Y-%m-%d-%X", time.localtime()) + (
        "" if args.exp_name is None else f"-{args.exp_name}"))
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    logging_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logging_dir, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

            with open(os.path.join(args.output_dir, "cmd.sh"), "w") as f:
                f.write("python " + " ".join(sys.argv))

            src_path = os.path.join(args.output_dir, 'src')
            os.makedirs(src_path, exist_ok=True)
            os.system(f"rsync -rv --exclude-from=.gitignore . {src_path}")

    accelerator.wait_for_everyone()

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    train_dataloader, eval_dataloader = get_dataloaders(args)
    tokenizer, vocab_size = get_tokenizer(args)

    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
        if args.model_type == "llama" and args.llama_attn_drop is not None:
            config.attention_dropout = args.llama_attn_drop
    else:
        assert False
    config.vocab_size = vocab_size
    if args.reward_prediction:
        config.output_hidden_states = True

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("gradient checkpointing enabled")

    if args.action_conditioned:
        # TODO: magic number
        perlude_tokens_num = (256 + 1) * args.context_length - 1
        tokens_per_dyna = 16
        model = HeadModelWithAction(model, action_dim=args.action_dim, prelude_tokens_num=perlude_tokens_num,
                                    tokens_num_per_dyna=tokens_per_dyna, context=args.context_length,
                                    segment_length=args.segment_length, model_type=args.model_type,
                                    reward_prediction=args.reward_prediction, action_recon=args.action_recon)

    if args.pretrained_transformer_path is not None:
        state_dict = load_file(os.path.join(args.pretrained_transformer_path, 'model.safetensors'))
        if args.load_internal_llm:
            model.llm.load_state_dict(state_dict, strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)
        logger.info("Finetuning the model from " + args.pretrained_transformer_path)
    else:
        logger.info("Training new model from scratch")

    if args.lora:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                 inference_mode=False,
                                 r=args.lora_r,
                                 lora_alpha=args.lora_alpha,
                                 lora_dropout=args.lora_dropout,
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                 "up_proj", "down_proj", "embed_tokens", "lm_head"],  # ! only for llama
                                 )
        if args.action_conditioned:
            model.llm = get_peft_model(model.llm, peft_config)
        else:
            model = get_peft_model(model, peft_config)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "layer_norm.weight"]
    no_decay = []
    if args.embed_no_wd:
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                if pn.endswith('bias') or \
                    (pn.endswith('weight') and isinstance(m, torch.nn.Embedding)) or \
                    (pn.endswith('weight') and isinstance(m, torch.nn.LayerNorm)) or \
                        (pn.endswith('weight') and isinstance(m, LlamaRMSNorm)):
                    fpn = '%s.%s' % (mn, pn) if mn else pn
                    no_decay.append(fpn)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    evaluator = Evaluator(args.i3d_path, max_batchsize=args.max_decode_batchsize)

    # Prepare everything with our `accelerator`.
    # we do not need to prepare train dataloader
    model, tokenizer, optimizer, lr_scheduler, evaluator, eval_dataloader = accelerator.prepare(
        model, tokenizer, optimizer, lr_scheduler, evaluator, eval_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    if args.start_completed_steps is not None:
        completed_steps = args.start_completed_steps
        progress_bar.update(completed_steps)
    starting_epoch = 0
    end = time.time()

    lastest_output_dir, lastest_completed_steps = None, None
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            raise NotImplementedError
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            # resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            resume_step = int(training_difference.replace("checkpoint_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

        lastest_output_dir, lastest_completed_steps = args.resume_from_checkpoint, completed_steps

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    avg_loss = None

    if args.eval_only:
        eval_logs = evaluate(args, accelerator, tokenizer, model, eval_dataloader, evaluator, completed_steps)
        if eval_logs is not None:
            print(args.pretrained_model_name_or_path)
            print(args.pretrained_transformer_path)
            print(eval_logs)
        return

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            print("skip first batches", resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            if args.action_conditioned:
                pixel_values, actions = batch
                actions = actions.to(accelerator.device, non_blocking=True)
                pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
            else:
                pixel_values = batch.to(accelerator.device, non_blocking=True)

            optimizer.zero_grad()

            with torch.no_grad():
                tokens, labels = accelerator.unwrap_model(tokenizer).tokenize(pixel_values, args.context_length,
                                                                              #   special_token=args.special_token
                                                                              )
                # recon = accelerator.unwrap_model(tokenizer).detokenize(tokens, args.context_length, special_token=args.special_token) # for debug
                model_input = {
                    'input_ids': tokens,
                    'labels': labels,
                }
                if args.action_conditioned:
                    model_input['action'] = actions

            with accelerator.accumulate(model):
                if args.reward_prediction:
                    outputs, rewards = model(**model_input)
                else:
                    outputs = model(**model_input)
                loss = outputs.loss
                avg_loss = accelerator.gather(loss.repeat(args.per_device_train_batch_size)).float().mean()
                if args.action_recon:
                    avg_action_recon_loss = accelerator.gather(accelerator.unwrap_model(
                        model).action_recon_loss.repeat(args.per_device_train_batch_size)).float().mean()
                accelerator.backward(loss)

                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if accelerator.sync_gradients and accelerator.is_main_process:
                batch_time = time.time() - end
                progress_bar.set_postfix(batch_time=batch_time)
                end = time.time()
                # Log metrics
                if completed_steps % args.log_steps == 0:
                    logs = {
                        "batch_time": batch_time,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "loss": avg_loss.item(),
                    }
                    if args.action_recon:
                        logs.update({"action_recon_loss": avg_action_recon_loss.item()})
                    accelerator.log(logs, step=completed_steps)

                # Save model checkpoint
                if completed_steps % checkpointing_steps == 0 and avg_loss < 4.0:
                    output_dir = f"checkpoints/checkpoint_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    lastest_output_dir = output_dir
                    lastest_completed_steps = completed_steps
                    if args.latest_checkpoint_only:
                        latest_checkpoint_path = os.path.join(args.output_dir,
                                                              f"checkpoints/checkpoint_{completed_steps - checkpointing_steps}")
                        if os.path.exists(latest_checkpoint_path):
                            os.system(f"rm -rf {latest_checkpoint_path}")

            if accelerator.sync_gradients:
                # Validation
                if completed_steps == args.max_train_steps or (completed_steps % args.validation_steps == 1 and (completed_steps > 1 or not args.skip_first_val)):
                    evaluate(args, accelerator, tokenizer, model, eval_dataloader, evaluator, completed_steps)

                # if avg_loss > 4.0:
                #     accelerator.load_state(lastest_output_dir)
                #     progress_bar.update(lastest_completed_steps - completed_steps)
                #     completed_steps = lastest_completed_steps
                #     print(f"Encounter avg_loss {avg_loss}, load state from", lastest_output_dir)

            if completed_steps >= args.max_train_steps:
                break

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )


if __name__ == "__main__":
    start_train()
