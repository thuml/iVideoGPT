import argparse
import os
import random
import imageio
import numpy as np

import torch
from transformers import AutoModelForCausalLM

from ivideogpt.vq_model import CompressiveVQModel
from utils import NPZParser

device = 'cuda'


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help="path to pretrained model")
    parser.add_argument('--input_path', type=str, required=True, help="path to input npz file")
    parser.add_argument('--dataset_name', type=str, required=True, help="dataset name")
    parser.add_argument('--output_path', type=str, default='outputs', help="path to save predicted video")

    parser.add_argument("--context_length", type=int, default=2, help="number of init context frames")
    parser.add_argument("--segment_length", type=int, default=16,
                        help="number of frames in total, including context and future frames")
    parser.add_argument('--resolution', type=int, default=64, help="resolution of frames")

    parser.add_argument('--repeat_times', default=5, type=int, help="number of times to repeat prediction")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()
    return args


@torch.no_grad
def predict(args, tokenizer, model, input):
    # prepare inputs
    pixel_values = input.to(device, non_blocking=True).unsqueeze(0)

    tokens, labels = tokenizer.tokenize(pixel_values, args.context_length)
    gen_input = tokens[:, :args.context_length * (16 * 16 + 1)]  # TODO: magic number

    # predict future frames
    max_new_tokens = (1 + 4 * 4) * (args.segment_length - args.context_length) - 1
    gen_kwargs = {
        'do_sample': True,
        'temperature': 1.0,
        'top_k': 100,
        'max_new_tokens': max_new_tokens,
    }
    generated_tokens = model.generate(
        gen_input.repeat(args.repeat_times, 1),
        **gen_kwargs,
        pad_token_id=50256,  # this is out of vocabulary but suppressing warning
    )

    # generated_tokens will include gen_input
    recon_output = tokenizer.detokenize(generated_tokens, args.context_length)
    recon_output = recon_output.clamp(0.0, 1.0)

    # save predicted video
    save_path = args.output_path
    os.makedirs(save_path, exist_ok=True)
    for j in range(args.repeat_times):
        gt_frames = [(pixel_values[0, i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                     for i in range(pixel_values.shape[1])]
        recon_frames = [(recon_output[j, i].permute(1, 2, 0).detach().cpu().numpy() *
                        255).astype(np.uint8) for i in range(recon_output.shape[1])]
        frames = [np.concatenate([gt_frames[i], recon_frames[i]], axis=1) for i in range(len(gt_frames))]
        imageio.mimsave(f"{save_path}/pred-samples-{j}.gif", frames, fps=4, loop=0)


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    # Load pretrained model and tokenizer
    tokenizer = CompressiveVQModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='tokenizer', low_cpu_mem_usage=False).to(device)
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='transformer', low_cpu_mem_usage=False).to(device)
    assert args.context_length == tokenizer.context_length
    assert model.config.vocab_size == tokenizer.num_vq_embeddings + tokenizer.num_dyn_embeddings + 2

    npz_parser = NPZParser(args.segment_length, args.resolution)
    input = npz_parser.parse(args.input_path, args.dataset_name)

    predict(args, tokenizer, model, input)


if __name__ == "__main__":
    main()
