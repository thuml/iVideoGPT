import argparse
import json
import sys
import os
import cv2
import time
from pathlib import Path
import psutil

import PIL
import PIL.Image
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
import numpy as np
from safetensors import safe_open

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available

from ivideogpt.vq_model import CompressiveVQModel, Discriminator, LPIPS
from ivideogpt.data import *


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_layer_wrt_loss(loss, layer):
    return torch.autograd.grad(
        outputs=loss,
        inputs=layer,
        grad_outputs=torch.ones_like(loss),
        retain_graph=True,
    )[0].detach()


def gradient_penalty(images, output, weight=10):
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    bsz = gradients.shape[0]
    gradients = torch.reshape(gradients, (bsz, -1))
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def save_checkpoint(model, discriminator, args, accelerator, global_step):
    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    discr_state_dict = accelerator.get_state_dict(discriminator)

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        torch.save(discr_state_dict, save_path / "unwrapped_discriminator")
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    accelerator.save_state(save_path)

    if args.latest_checkpoint_only:
        latest_checkpoint_path = Path(args.output_dir) / f"checkpoint-{global_step-args.checkpointing_steps}"
        if accelerator.is_main_process:
            if latest_checkpoint_path.exists():
                os.system(f"rm -rf {latest_checkpoint_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--log_grad_norm_steps", type=int, default=500,
                        help=("Print logs of gradient norms every X steps."))
    parser.add_argument("--log_steps", type=int, default=50, help=("Print logs every X steps."))
    parser.add_argument("--validation_steps", type=int, default=5000,
                        help=(
                            "Run validation every X steps. Validation consists of running reconstruction on images in"
                            " `args.validation_images` and logging the reconstructed images."
                        ),
                        )
    parser.add_argument("--log_image_steps", type=int, default=100)
    parser.add_argument("--vae_loss", type=str, default="l1", help="The loss function for vae reconstruction loss.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--model_config_name_or_path", type=str, default=None,
                        help="The config of the Vq model to train, leave as None to use standard DDPM configuration.")
    parser.add_argument("--discriminator_config_name_or_path", type=str, default=None,
                        help="The config of the discriminator model to train, leave as None to use standard DDPM configuration.")
    parser.add_argument("--dataset_name", type=str, default="robotic",
                        help=(
                            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
                            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
                            " or to a folder containing files that 🤗 Datasets can understand."
                        ),
                        )
    parser.add_argument("--output_dir", type=str, default="vqgan-output",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="The directory where the downloaded models and datasets will be stored.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=256,
                        help=(
                            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                            " resolution"
                        ),
                        )
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000000,
                        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--discr_learning_rate", type=float, default=1e-4,
                        help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False,
                        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--discr_lr_scheduler", type=str, default="constant_with_warmup",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help=(
                            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
                        ),
                        )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help=(
                            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                        help=(
        "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--checkpointing_steps", type=int, default=5000,
                        help=(
                            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
                            " training using `--resume_from_checkpoint`."
                        ),
                        )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help=(
                            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
                            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
                        ),
                        )
    parser.add_argument("--enable_xformers_memory_efficient_attention",
                        action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--tracker_project_name", type=str, default="vqgan-training",
                        help=(
                            "The `project_name` argument passed to Accelerator.init_trackers for"
                            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
                        ),
                        )
    parser.add_argument("--segment_length", type=int, default=5,
                        help="The length of the segmented trajectories to use for the training.")
    parser.add_argument("--context_length", type=int, default=1)
    parser.add_argument("--segment_horizon", type=int, default=16)
    parser.add_argument('--video_stepsize', default=1, type=int)
    parser.add_argument('--rand_select', default=False, action='store_true')
    parser.add_argument('--rand_shuffle', default=False, action='store_true')
    parser.add_argument('--model_type', default='vqgan', type=str, choices=['vqgan', 'ctx_vqgan'], help='Type of model to use')
    parser.add_argument('--dataset_path', default='/data2/tensorflow_datasets',
                        type=str, help='Path to the tensorflow datasets')
    parser.add_argument('--dataset_size', default=None, type=int)
    parser.add_argument('--weighted_mse', default=None, type=float)
    parser.add_argument('--weighted_gan', default=False, action='store_true')
    parser.add_argument('--disc_start', default=0, type=int)
    parser.add_argument('--disc_weight', default=0.8, type=float)
    parser.add_argument('--latest_checkpoint_only', default=False, action='store_true')
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--disc_depth', default=4, type=int)
    parser.add_argument('--perc_weight', default=1.0, type=float)
    parser.add_argument('--recon_weight', default=1.0, type=float)
    parser.add_argument('--oxe_data_mixes_type', default='frac', type=str)
    parser.add_argument('--strong_aug', default=False, action='store_true')
    parser.add_argument('--sthsth_root_path',
                        default='/data/something-something-v2/20bn-something-something-v2-frames-64', type=str)
    parser.add_argument('--skip_first_val', default=False, action='store_true')
    parser.add_argument('--start_global_step', default=0, type=int)
    parser.add_argument('--balanced_loss', default=False, action='store_true')
    parser.add_argument('--selected_params', default=False, action='store_true')
    parser.add_argument('--no_aug', default=False, action='store_true')

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def plot_img(img, postfix=''):
    cv2.imwrite(f'tmp-img{postfix}.png', img[0].detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255)


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    args = parse_args()
    args.output_dir = os.path.join(args.output_dir, time.strftime(
        "%Y-%m-%d-%X", time.localtime()) + ("" if args.exp_name is None else f"-{args.exp_name}"))
    os.makedirs(args.output_dir, exist_ok=True)

    # Enable TF32 on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(logging_dir, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

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

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    if args.model_config_name_or_path is None and args.pretrained_model_name_or_path is None:
        if args.model_type == "ctx_vqgan":
            if args.resolution == 64:
                config = json.load(open("configs/ctx_vae64/config.json"))
                config.update({"context_length": args.context_length})
                model = CompressiveVQModel(**config)
            elif args.resolution == 256:
                config = json.load(open("configs/ctx_vae/config.json"))
                model = CompressiveVQModel(**config)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            config = json.load(open("configs/vae/config.json"))
            model = VQModel(**config)
    elif args.pretrained_model_name_or_path is not None:
        if args.model_type == "ctx_vqgan":
            model = CompressiveVQModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder=None, revision=None, variant=None, use_safetensor=True,
                low_cpu_mem_usage=False, device_map=None,
                ignore_mismatched_sizes=True
            )
            if args.pretrained_model_name_or_path == "pretrained_models/amused/vqvae":
                model.init_modules()
            if args.context_length != model.context_length:
                print(
                    f"[Warning] pretrained context length mismatch, change from {model.context_length} to {args.context_length}")
                model.set_context_length(args.context_length)
        elif args.model_type == "vqgan":
            raise NotImplementedError
            model = VQModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder=None, revision=None, variant=None, use_safetensor=True
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
        config = VQModel.load_config(args.model_config_name_or_path)
        model = VQModel.from_config(config)

    if args.use_ema:
        ema_model = EMAModel(model.parameters(), model_cls=VQModel, model_config=model.config)
    if args.discriminator_config_name_or_path is None:
        discriminator = Discriminator(depth=args.disc_depth)
    else:
        discriminator = Discriminator(depth=args.disc_depth)
        discriminator.load_state_dict(torch.load(args.discriminator_config_name_or_path))

    # Perceptual loss
    lpips = LPIPS().to(accelerator.device).eval()
    # Enable flash attention if asked
    if args.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()

    learning_rate = args.learning_rate
    if args.scale_lr:
        learning_rate = (
            learning_rate * args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if args.selected_params:
        # frozon codebook params
        if args.model_type == 'ctx_vqgan':
            params = [parameter for name, parameter in model.named_parameters() if 'quantize' not in name]
        else:
            raise NotImplementedError
    else:
        params = list(model.parameters())
    optimizer = optimizer_cls(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    discr_optimizer = optimizer_cls(
        list(discriminator.parameters()),
        lr=args.discr_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    ##################################
    # DATLOADER and LR-SCHEDULER     #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # DataLoaders creation:

    if args.dataset_name != "robotic":
        raise NotImplementedError

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
        'random_selection': args.rand_select,
        'random_shuffle': args.rand_shuffle,
        'goal_conditioned': False,
        'segment_length': args.segment_length,
        'context_length': args.context_length,
        'stepsize': args.video_stepsize,
        'segment_horizon': args.segment_horizon,
    }
    train_dataloader = SimpleRoboticDataLoaderv2(
        parent_dir=args.dataset_path,
        datasets=DATASET_NAMED_MIXES[args.oxe_data_mixes_type],
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        train=True,
        maxsize=args.dataset_size,
        image_size=args.resolution,
        sthsth_root_path=args.sthsth_root_path,
        **augmentation_args,
        **segment_args,
    )
    eval_dataloader = SimpleRoboticDataLoaderv2(
        parent_dir=args.dataset_path,
        datasets=DATASET_NAMED_MIXES[args.oxe_data_mixes_type],
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        train=False,
        image_size=args.resolution,
        sthsth_root_path=args.sthsth_root_path,
        **augmentation_args,
        **segment_args,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    )
    discr_lr_scheduler = get_scheduler(
        args.discr_lr_scheduler,
        optimizer=discr_optimizer,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    )

    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model, discriminator, optimizer, discr_optimizer, lr_scheduler, discr_lr_scheduler = accelerator.prepare(
        model, discriminator, optimizer, discr_optimizer, lr_scheduler, discr_lr_scheduler
    )
    # Train!
    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = args.start_global_step
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            path = os.path.join(args.output_dir, path)

        if path is None:
            accelerator.print(f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            accelerator.wait_for_everyone()
            global_step = int(os.path.basename(path).split("-")[1])
            # first_epoch = global_step // num_update_steps_per_epoch

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    avg_gen_loss, avg_discr_loss = None, None
    avg_recon_loss, avg_commit_loss, avg_dyna_commit_loss, avg_perceptual_loss, avg_gan_loss, adaptive_weight, avg_residual_loss, avg_flow_loss = None, None, None, None, None, None, None, None
    avg_ref_recon_loss, avg_ref_perceptual_loss = None, None
    avg_feat_loss = None
    avg_fake_logits, avg_real_logits = None, None

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            pixel_values = batch.to(accelerator.device, non_blocking=True)
            pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
            original_pixel_values = pixel_values

            generator_step = ((i // args.gradient_accumulation_steps) % 2) == 0

            if generator_step:
                data_time_m.update(time.time() - end)
            # Train Step
            # The behavior of accelerator.accumulate is to
            # 1. Check if gradients are synced(reached gradient-accumulation_steps)
            # 2. If so sync gradients by stopping the not syncing process
            if generator_step:
                optimizer.zero_grad(set_to_none=True)
            else:
                discr_optimizer.zero_grad(set_to_none=True)
            # encode images to the latent space and get the commit loss from vq tokenization
            # Return commit loss

            if generator_step or global_step >= args.disc_start:
                if 'ctx' in args.model_type:
                    with torch.no_grad():
                        BT, C, H, W = pixel_values.shape
                        B, T = (BT // args.segment_length), args.segment_length
                        frame_pixel_values = pixel_values.reshape(
                            args.train_batch_size, args.segment_length, C, H, W)  # B, T, C, H, W
                        target = frame_pixel_values[:, args.context_length:].reshape(
                            B * (T - args.context_length), C, H, W)  # B*(T-t), C, H, W
                        if args.context_length > 1:
                            reference_single = frame_pixel_values[:,
                                                                  :args.context_length].reshape(-1, C, H, W)  # B*t, C, H, W
                            reference = None  # to raise not implementation error
                        else:
                            reference = frame_pixel_values[:, args.context_length - 1:args.context_length].repeat(
                                1, args.segment_length - args.context_length, 1, 1, 1).reshape(B * (T - args.context_length), C, H, W)  # B*(T-t), C, H, W
                            reference_single = frame_pixel_values[:, args.context_length - 1]
                        pixel_values = target

                    if args.model_type == 'ctx_vqgan':
                        fmap, fmap_ref, commit_loss, dyna_commit_loss = model(sample=reference_single,
                                                                                     dyn_sample=target,
                                                                                     return_dict=False,
                                                                                     return_loss=True,
                                                                                     segment_len=args.segment_length - args.context_length)
                    else:
                        raise NotImplementedError
                else:
                    fmap, commit_loss = model(pixel_values, return_dict=False, return_loss=True)

                # weights for weighted losses
                weights = None
                weights_single = None

            if generator_step:
                with accelerator.accumulate(model):
                    def avg_loss(loss):
                        return accelerator.gather(loss.repeat(args.train_batch_size)).float().mean()

                    # reconstruction loss. Pixel level differences between input vs output
                    def get_recon_loss(gt, recon, weights):
                        if args.vae_loss == "l2":
                            loss = F.mse_loss(gt, recon, reduction='none')
                        else:
                            loss = F.l1_loss(gt, recon, reduction='none')
                        if weights is not None:
                            resized_weights = F.interpolate(weights, loss.shape[2:])
                            loss = (loss * resized_weights).mean()
                        else:
                            loss = loss.mean()
                        return loss

                    recon_loss = get_recon_loss(pixel_values, fmap, weights)
                    if args.balanced_loss:
                        loss = args.recon_weight * recon_loss * \
                            (args.segment_length - args.context_length) / args.segment_length
                    else:
                        loss = args.recon_weight * recon_loss
                    avg_recon_loss = avg_loss(recon_loss)

                    if 'ctx' in args.model_type:
                        ref_recon_loss = get_recon_loss(reference_single, fmap_ref, weights_single)
                        if args.balanced_loss:
                            loss += args.recon_weight * ref_recon_loss * args.context_length / args.segment_length
                        else:
                            loss += args.recon_weight * ref_recon_loss
                        avg_ref_recon_loss = avg_loss(ref_recon_loss)

                    # perceptual loss. The high level feature mean squared error loss
                    perceptual_loss = lpips(
                        pixel_values.contiguous() * 2 - 1.0,
                        fmap.contiguous() * 2 - 1.0,
                        weight=weights
                    ).mean()
                    if args.balanced_loss:
                        loss += args.perc_weight * perceptual_loss * \
                            (args.segment_length - args.context_length) / args.segment_length
                    else:
                        loss += args.perc_weight * perceptual_loss
                    avg_perceptual_loss = avg_loss(perceptual_loss)
                    if 'ctx' in args.model_type:
                        ref_perceptual_loss = lpips(
                            reference_single.contiguous() * 2 - 1.0,
                            fmap_ref.contiguous() * 2 - 1.0,
                            weight=weights
                        ).mean()
                        if args.balanced_loss:
                            loss += args.perc_weight * ref_perceptual_loss * args.context_length / args.segment_length
                        else:
                            loss += args.perc_weight * ref_perceptual_loss
                        avg_ref_perceptual_loss = avg_loss(ref_perceptual_loss)

                    # generator loss
                    if global_step >= args.disc_start:
                        if 'ctx' in args.model_type:
                            disc_fmap = torch.cat([fmap_ref, fmap], dim=0)
                            if weights_single is None or weights is None:
                                disc_weights = None
                            else:
                                disc_weights = torch.cat([weights_single, weights], dim=0)
                        else:
                            disc_fmap = fmap
                            disc_weights = weights

                        if disc_weights is not None and args.weighted_gan:
                            logits = discriminator(disc_fmap)
                            resized_weights = F.interpolate(disc_weights, logits.shape[2:])
                            gen_loss = -(resized_weights * logits).mean()
                        else:
                            gen_loss = -discriminator(disc_fmap).mean()

                        last_dec_layer = accelerator.unwrap_model(model).cond_decoder.conv_out.weight
                        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p=2)
                        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p=2)

                        adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min=1e-8)
                        adaptive_weight = adaptive_weight.clamp(max=1e4)

                        loss += args.disc_weight * adaptive_weight * gen_loss
                        avg_gan_loss = avg_loss(gen_loss)

                    # regularization losses
                    loss += commit_loss
                    avg_commit_loss = avg_loss(commit_loss)
                    if 'ctx' in args.model_type:
                        loss += dyna_commit_loss
                        avg_dyna_commit_loss = avg_loss(dyna_commit_loss)

                    # Gather thexd losses across all processes for logging (if we use distributed training).
                    avg_gen_loss = avg_loss(loss)

                    accelerator.backward(loss)

                    # print("detect unused_parameters for debug")
                    # for name, param in model.named_parameters():
                    #     if param.grad is None:
                    #         print(name)

                    if args.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    # log gradient norm before zeroing it
                    if accelerator.sync_gradients and global_step % args.log_grad_norm_steps == 0 and accelerator.is_main_process:
                        log_grad_norm(model, accelerator, global_step)
            else:
                # Return discriminator loss
                with accelerator.accumulate(discriminator):
                    if global_step >= args.disc_start:
                        fmap.detach_()
                        if 'ctx' in args.model_type:
                            fmap_ref.detach_()
                        # pixel_values.requires_grad_()

                        if 'ctx' in args.model_type:
                            disc_pixel_values = torch.cat([reference_single, pixel_values], dim=0)
                            disc_fmap = torch.cat([fmap_ref, fmap], dim=0)
                            if weights_single is None or weights is None:
                                disc_weights = None
                            else:
                                disc_weights = torch.cat([weights_single, weights], dim=0)
                        else:
                            disc_pixel_values = pixel_values
                            disc_fmap = fmap
                            disc_weights = weights

                        real = discriminator(disc_pixel_values)
                        fake = discriminator(disc_fmap)
                        if weights is not None and args.weighted_gan:
                            resized_weights = F.interpolate(disc_weights, fake.shape[2:])
                            loss = (resized_weights * F.relu(1 + fake) + resized_weights * F.relu(1 - real)).mean()
                        else:
                            loss = (F.relu(1 + fake) + F.relu(1 - real)).mean()
                        # gp = gradient_penalty(pixel_values, real)
                        # loss += gp

                        if global_step < args.disc_start:
                            loss = loss * 0.0

                        avg_discr_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                        if weights is not None and args.weighted_gan:
                            avg_fake_logits = accelerator.gather(
                                (resized_weights * fake).mean().repeat(args.train_batch_size)).mean()
                            avg_real_logits = accelerator.gather(
                                (resized_weights * real).mean().repeat(args.train_batch_size)).mean()
                        else:
                            avg_fake_logits = accelerator.gather(fake.mean().repeat(args.train_batch_size)).mean()
                            avg_real_logits = accelerator.gather(real.mean().repeat(args.train_batch_size)).mean()
                        accelerator.backward(loss)

                        if args.max_grad_norm is not None and accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)

                        discr_optimizer.step()
                        discr_lr_scheduler.step()
                        if accelerator.sync_gradients and global_step % args.log_grad_norm_steps == 0 and accelerator.is_main_process:
                            log_grad_norm(discriminator, accelerator, global_step)
                    else:
                        pass  # skip discriminator step if not started
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if args.use_ema:
                    ema_model.step(model.parameters())
            if accelerator.sync_gradients and not generator_step and accelerator.is_main_process:
                # wait for both generator and discriminator to settle
                batch_time_m.update(time.time() - end)
                progress_bar.set_postfix(batch_time=batch_time_m.val, data_time=data_time_m.val)
                end = time.time()
                # Log metrics
                if global_step % args.log_steps == 0:
                    samples_per_second_per_gpu = (
                        args.gradient_accumulation_steps * args.train_batch_size / batch_time_m.val
                    )
                    logs = {
                        "step_discr_loss": avg_discr_loss.item() if avg_discr_loss is not None else 0.0,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    if avg_gen_loss is not None:
                        logs["step_gen_loss"] = avg_gen_loss.item()
                    if avg_recon_loss is not None:
                        logs["gen_loss/step_recon_loss"] = avg_recon_loss.item()
                    if avg_ref_recon_loss is not None:
                        logs["gen_loss/step_ref_recon_loss"] = avg_ref_recon_loss.item()
                    if avg_commit_loss is not None:
                        logs["gen_loss/step_commit_loss"] = avg_commit_loss.item()
                    if avg_dyna_commit_loss is not None:
                        logs["gen_loss/step_dyna_commit_loss"] = avg_dyna_commit_loss.item()
                    if avg_perceptual_loss is not None:
                        logs["gen_loss/step_perceptual_loss"] = avg_perceptual_loss.item()
                    if avg_ref_perceptual_loss is not None:
                        logs["gen_loss/step_ref_perceptual_loss"] = avg_ref_perceptual_loss.item()
                    if avg_gan_loss is not None:
                        logs["gen_loss/step_gan_loss"] = avg_gan_loss.item()
                    if adaptive_weight is not None:
                        logs["gen_loss/adaptive_weight"] = adaptive_weight.item()
                    if avg_fake_logits is not None:
                        logs["disc_loss/step_fake_logits"] = avg_fake_logits.item()
                    if avg_real_logits is not None:
                        logs["disc_loss/step_real_logits"] = avg_real_logits.item()
                    if avg_fake_logits is not None and avg_real_logits is not None:
                        logs["disc_loss/step_logit_diff"] = avg_real_logits.item() - avg_fake_logits.item()
                    if avg_residual_loss is not None:
                        logs["gen_loss/step_residual_loss"] = avg_residual_loss.item()
                    if avg_flow_loss is not None:
                        logs["gen_loss/step_flow_loss"] = avg_flow_loss.item()
                    if avg_feat_loss is not None:
                        logs["gen_loss/step_feat_loss"] = avg_feat_loss.item()

                    logs["mem_used"] = psutil.virtual_memory().used / 1024 / 1024 / 1024
                    accelerator.log(logs, step=global_step)

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()
                # Save model checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(model, discriminator, args, accelerator, global_step)

            if accelerator.sync_gradients and generator_step and accelerator.is_main_process:
                # Generate images
                if global_step % args.log_image_steps == 1:
                    with torch.no_grad():
                        save_path = os.path.join(args.output_dir, "images", f"train-samples-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        segment_length = args.segment_length - args.context_length

                        np_img = lambda x: x.detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255

                        gt = np.concatenate([np_img(pixel_values[i]) for i in range(segment_length)], 1)
                        recon = np.concatenate([np_img(fmap[i]) for i in range(segment_length)], 1)
                        if 'ctx' in args.model_type:
                            if args.context_length > 1:
                                # fmap_ref: B, t, C, H, W -> B, C, H, t*W
                                ref_recon = fmap_ref.reshape(B, args.context_length, C, H, W)[0].permute(
                                    1, 2, 0, 3).reshape(C, H, args.context_length * W)
                                ref_recon = np_img(ref_recon)
                            else:
                                ref_recon = np_img(fmap_ref[0])
                        diff = np.concatenate([np_img(fmap[i] - fmap[max(i - 1, 0)]) for i in range(segment_length)], 1)
                        # error = np.concatenate([np_img(torch.abs(fmap[i] - pixel_values[i])) for i in range(segment_length)], 1)

                        context = np.concatenate([np_img(original_pixel_values[i])
                                                 for i in range(args.context_length)], 1)
                        gt = np.concatenate([context, gt], 1)
                        recon = np.concatenate([np.zeros_like(context), recon], 1)
                        if 'ctx' in args.model_type:
                            if args.context_length > 1:
                                recon[:, :args.context_length * args.resolution, :] = ref_recon
                            else:
                                recon[:, (args.context_length - 1) * args.resolution:args.context_length *
                                      args.resolution, :] = ref_recon
                        diff = np.concatenate([np.zeros_like(context), diff], 1)
                        # error = np.concatenate([np.zeros_like(context), error], 1)
                        error = np.abs(recon - gt)

                        cv2.imwrite(os.path.join(
                            save_path, f'train-samples-{global_step}.png'), np.concatenate([gt, recon, diff, error], 0))

                # Validation
                if global_step % args.validation_steps == 1 and (global_step > 1 or not args.skip_first_val):
                    with torch.no_grad():
                        model.eval()
                        recon_losses = []
                        perceptual_losses = []
                        val_iters = 100
                        bar = tqdm(range(val_iters), desc="validation")

                        for i, batch in enumerate(eval_dataloader):
                            if i == val_iters:
                                break

                            # preprocess
                            pixel_values = batch.to(accelerator.device, non_blocking=True)
                            pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
                            original_pixel_values = pixel_values

                            BT, C, H, W = pixel_values.shape
                            B, T = (BT // args.segment_length), args.segment_length
                            frame_pixel_values = pixel_values.reshape(
                                args.train_batch_size, args.segment_length, C, H, W)  # B, T, C, H, W
                            target = frame_pixel_values[:, args.context_length:].reshape(
                                B * (T - args.context_length), C, H, W)  # B*(T-t), C, H, W
                            if args.context_length > 1:
                                # B*t, C, H, W
                                reference_single = frame_pixel_values[:, :args.context_length].reshape(-1, C, H, W)
                                reference = None  # to raise not implementation error
                            else:
                                reference = frame_pixel_values[:, args.context_length - 1:args.context_length].repeat(
                                    1, args.segment_length - args.context_length, 1, 1, 1).reshape(B * (T - args.context_length), C, H, W)  # B*(T-t), C, H, W
                                reference_single = frame_pixel_values[:, args.context_length - 1]
                            pixel_values = target

                            # compute weights
                            weights = None
                            weights_single = None

                            # compute losses
                            if args.model_type == 'ctx_vqgan':
                                fmap, fmap_ref, commit_loss, dyna_commit_loss = model(sample=reference_single,
                                                                                             dyn_sample=target,
                                                                                             return_dict=False,
                                                                                             return_loss=True,
                                                                                             segment_len=args.segment_length - args.context_length)
                            else:
                                fmap, commit_loss = model(pixel_values, return_dict=False, return_loss=True)

                            recon_loss = get_recon_loss(pixel_values, fmap, weights)
                            perceptual_loss = lpips(
                                pixel_values.contiguous() * 2 - 1.0,
                                fmap.contiguous() * 2 - 1.0,
                                weight=weights
                            ).mean()
                            recon_losses.append(recon_loss)
                            perceptual_losses.append(perceptual_loss)

                            # log images
                            if i % 10 == 0:
                                save_path = os.path.join(args.output_dir, "images", f"val-samples-{global_step}")
                                os.makedirs(save_path, exist_ok=True)
                                segment_length = args.segment_length - args.context_length

                                np_img = lambda x: x.detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255

                                gt = np.concatenate([np_img(pixel_values[i]) for i in range(segment_length)], 1)
                                recon = np.concatenate([np_img(fmap[i]) for i in range(segment_length)], 1)
                                if 'ctx' in args.model_type:
                                    if args.context_length > 1:
                                        # fmap_ref: B, t, C, H, W -> B, C, H, t*W
                                        ref_recon = fmap_ref.reshape(B, args.context_length, C, H, W)[0].permute(
                                            1, 2, 0, 3).reshape(C, H, args.context_length * W)
                                        ref_recon = np_img(ref_recon)
                                    else:
                                        ref_recon = np_img(fmap_ref[0])
                                diff = np.concatenate([np_img(fmap[i] - fmap[max(i - 1, 0)])
                                                      for i in range(segment_length)], 1)
                                # error = np.concatenate([np_img(torch.abs(fmap[i] - pixel_values[i])) for i in range(segment_length)], 1)

                                context = np.concatenate([np_img(original_pixel_values[i])
                                                         for i in range(args.context_length)], 1)
                                gt = np.concatenate([context, gt], 1)
                                recon = np.concatenate([np.zeros_like(context), recon], 1)
                                if 'ctx' in args.model_type:
                                    if args.context_length > 1:
                                        recon[:, :args.context_length * args.resolution, :] = ref_recon
                                    else:
                                        recon[:, (args.context_length - 1) *
                                              args.resolution:args.context_length * args.resolution, :] = ref_recon
                                diff = np.concatenate([np.zeros_like(context), diff], 1)
                                # error = np.concatenate([np.zeros_like(context), error], 1)
                                error = np.abs(recon - gt)
                                cv2.imwrite(os.path.join(
                                    save_path, f'val-samples-{global_step}-{i}.png'), np.concatenate([gt, recon, diff, error], 0))

                            bar.update(1)

                        accelerator.log({
                            'val_loss/recon_loss': torch.stack(recon_losses).mean().item(),
                            'val_loss/perceptual_loss': torch.stack(perceptual_losses).mean().item(),
                        }, step=global_step)
                        model.train()

            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:
                break
        # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, discriminator, args, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if args.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
