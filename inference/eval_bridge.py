"""Evaluate the Bridge V2 fine-tuned action-conditioned iVideoGPT on held-out episodes.

Protocol: for every episode in data/bridge_npz/val, take --segments_per_episode
evenly-spaced 16-frame segments (stepsize 1, matching fine-tuning). Condition on
2 context frames (+ per-step 7-dim actions for the action-conditioned model),
autoregressively predict the remaining 14 frames, and score predictions against
ground truth with PSNR / SSIM / LPIPS (predicted frames only) and FVD (full clips).

Compares two models sharing the same frozen tokenizer:
  - finetuned: HeadModelWithAction from --run_dir (transformer/ + tokenizer/)
  - baseline:  the OXE-pretrained action-free transformer (zero-shot on Bridge)

Generation runs on MPS/CUDA; all metrics run on CPU (piqa/lpips/i3d are not
reliable on MPS).

Usage (from repo root):
  PYTHONPATH=.:inference .venv/bin/python -u inference/eval_bridge.py \
      --run_dir log_trm/2026-07-10-16:11:47-bridge_act_cond_run
"""
import argparse
import json
import os
import time

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from ivideogpt.vq_model import CompressiveVQModel
from ivideogpt.transformer import HeadModelWithAction
from ivideogpt.utils.video_metric import Evaluator, FeatureStats

GEN_DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
TOKENS_PER_CONTEXT_FRAME = 16 * 16 + 1   # 257, matches inference/predict.py
TOKENS_PER_DYNA_FRAME = 1 + 4 * 4        # 17


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', type=str, required=True, help='fine-tune output dir with tokenizer/ and transformer/')
    p.add_argument('--baseline_path', type=str, default='pretrained_models/ivideogpt-oxe-64-act-free',
                   help='OXE-pretrained act-free model dir; pass "none" to skip the baseline')
    p.add_argument('--val_dir', type=str, default='data/bridge_npz/val')
    p.add_argument('--i3d_path', type=str, default='pretrained_models/i3d/i3d_torchscript.pt')
    p.add_argument('--out_dir', type=str, default=None, help='default: <run_dir>/eval_bridge')
    p.add_argument('--segments_per_episode', type=int, default=2)
    p.add_argument('--segment_length', type=int, default=16)
    p.add_argument('--context_length', type=int, default=2)
    p.add_argument('--resolution', type=int, default=64)
    p.add_argument('--action_dim', type=int, default=7)
    p.add_argument('--num_gifs', type=int, default=4)
    p.add_argument('--limit_episodes', type=int, default=None, help='smoke-test cap')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def load_episode_segments(npz_path, seg_len, segs_per_ep, resolution):
    d = np.load(npz_path)
    frames, actions = d['image'], d['action']
    n = len(frames)
    if n < seg_len:
        return []
    n_segs = min(segs_per_ep, n - seg_len + 1)
    starts = sorted(set(int(s) for s in np.linspace(0, n - seg_len, n_segs)))
    out = []
    for s in starts:
        img = torch.from_numpy(frames[s:s + seg_len]).float().permute(0, 3, 1, 2) / 255.0
        img = TF.resize(img, [resolution, resolution]).clamp(0.0, 1.0)  # resize interpolation can overshoot [0,1]
        act = torch.from_numpy(actions[s:s + seg_len]).float()
        out.append((img, act, s))
    return out


def load_finetuned(run_dir, args):
    # NOTE: the ctx_vqgan DECODER is numerically broken on MPS (tokenize -> identical
    # token ids on CPU/MPS, but detokenize -> ~14 PSNR mush vs ~24-31 PSNR on CPU;
    # transformer logits match to 1e-6). So: tokenize + generate on GEN_DEVICE,
    # detokenize on a CPU copy of the tokenizer.
    tokenizer = CompressiveVQModel.from_pretrained(run_dir, subfolder='tokenizer', low_cpu_mem_usage=False).to(GEN_DEVICE).eval()
    config = AutoConfig.from_pretrained(run_dir, subfolder='transformer')
    llm = AutoModelForCausalLM.from_config(config)
    model = HeadModelWithAction(llm, action_dim=args.action_dim,
                                prelude_tokens_num=TOKENS_PER_CONTEXT_FRAME * args.context_length - 1,
                                tokens_num_per_dyna=16,
                                context=args.context_length,
                                segment_length=args.segment_length)
    model.load_state_dict(load_file(os.path.join(run_dir, 'transformer', 'model.safetensors')), strict=True)
    return tokenizer, model.to(GEN_DEVICE).eval()


@torch.no_grad()
def generate(tokenizer, tokenizer_cpu, model, pixel_values, actions, args):
    """pixel_values: (1,T,C,H,W) on GEN_DEVICE; actions: (1,T,A) or None. Returns (1,T,C,H,W) in [0,1] on CPU."""
    tokens, _ = tokenizer.tokenize(pixel_values, args.context_length)
    gen_input = tokens[:, :args.context_length * TOKENS_PER_CONTEXT_FRAME]
    max_new_tokens = TOKENS_PER_DYNA_FRAME * (args.segment_length - args.context_length) - 1
    extra = {'action': actions} if actions is not None else {'attention_mask': torch.ones_like(gen_input)}
    generated = model.generate(gen_input, do_sample=True, temperature=1.0, top_k=100,
                               max_new_tokens=max_new_tokens, pad_token_id=50256, **extra)
    return tokenizer_cpu.detokenize(generated.cpu(), args.context_length).clamp(0.0, 1.0)


@torch.no_grad()
def i3d_features(evaluator, clips, batch=8):
    """clips: (N,T,C,H,W) in [0,1] on CPU -> FeatureStats"""
    stats = FeatureStats(capture_mean_cov=True)
    kw = dict(rescale=True, resize=True, return_features=True)
    for i in range(0, clips.shape[0], batch):
        x = clips[i:i + batch].permute(0, 2, 1, 3, 4).contiguous() * 255.0  # N,C,T,H,W
        stats.append_torch(evaluator.i3d_model(x, **kw))
    return stats


def to_gif_frames(*videos):
    """each video: (T,C,H,W) [0,1] cpu -> list of HxW*len(videos) uint8 frames"""
    T = videos[0].shape[0]
    return [np.concatenate([(v[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for v in videos], axis=1)
            for t in range(T)]


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.run_dir, 'eval_bridge')
    os.makedirs(os.path.join(out_dir, 'gifs'), exist_ok=True)

    files = sorted(f for f in os.listdir(args.val_dir) if f.endswith('.npz'))
    if args.limit_episodes:
        files = files[:args.limit_episodes]
    segments = []  # (episode_file, start, img (T,C,H,W), act (T,A))
    for f in files:
        for img, act, s in load_episode_segments(os.path.join(args.val_dir, f),
                                                 args.segment_length, args.segments_per_episode, args.resolution):
            segments.append((f, s, img, act))
    print(f'{len(files)} episodes -> {len(segments)} segments of {args.segment_length} frames', flush=True)

    models = {}
    tokenizer, finetuned = load_finetuned(args.run_dir, args)
    tokenizer_cpu = CompressiveVQModel.from_pretrained(
        args.run_dir, subfolder='tokenizer', low_cpu_mem_usage=False).eval()  # CPU decoder (see load_finetuned note)
    models['finetuned_act_cond'] = finetuned
    if args.baseline_path.lower() != 'none':
        baseline = AutoModelForCausalLM.from_pretrained(
            args.baseline_path, subfolder='transformer', low_cpu_mem_usage=False).to(GEN_DEVICE).eval()
        models['pretrained_act_free'] = baseline

    evaluator = Evaluator(i3d_path=args.i3d_path).eval()  # CPU

    ctx = args.context_length
    preds = {name: [] for name in models}   # cpu tensors (T,C,H,W)
    results = {name: [] for name in models}
    t0 = time.time()
    for idx, (fname, start, img, act) in enumerate(segments):
        torch.manual_seed(args.seed + idx)
        pv = img.unsqueeze(0).to(GEN_DEVICE)
        for name, model in models.items():
            actions = act.unsqueeze(0).to(GEN_DEVICE) if name == 'finetuned_act_cond' else None
            recon = generate(tokenizer, tokenizer_cpu, model, pv, actions, args)[0]
            preds[name].append(recon)
            mse, psnr, ssim, lp = evaluator(img[ctx:].unsqueeze(0), recon[ctx:].unsqueeze(0))
            results[name].append(dict(episode=fname, start=start, mse=float(mse), psnr=float(psnr),
                                      ssim=float(ssim), lpips=float(lp)))
        done, total = idx + 1, len(segments)
        eta = (time.time() - t0) / done * (total - done)
        print(f'[{done}/{total}] {fname} start={start} eta {eta/60:.1f} min', flush=True)

    # FVD over all clips per model
    gt_clips = torch.stack([s[2] for s in segments])
    real_stats = i3d_features(evaluator, gt_clips)
    summary = {}
    for name in models:
        gen_stats = i3d_features(evaluator, torch.stack(preds[name]))
        agg = {m: (float(np.mean([r[m] for r in results[name]])),
                   float(np.std([r[m] for r in results[name]])))
               for m in ('mse', 'psnr', 'ssim', 'lpips')}
        agg['fvd'] = evaluator.compute_fvd(real_stats, gen_stats)
        summary[name] = agg
        print(name, json.dumps(agg, indent=2), flush=True)

    # side-by-side GIFs: GT | finetuned | baseline, first segment of the first --num_gifs episodes
    made = set()
    for idx, (fname, start, img, _) in enumerate(segments):
        if fname in made or len(made) >= args.num_gifs:
            continue
        made.add(fname)
        videos = [img] + [preds[name][idx] for name in models]
        imageio.mimsave(os.path.join(out_dir, 'gifs', f'{fname.replace(".npz", "")}-gt-vs-pred.gif'),
                        to_gif_frames(*videos), fps=4, loop=0)

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({'config': vars(args), 'summary': summary, 'per_segment': results}, f, indent=2)

    with open(os.path.join(out_dir, 'metrics.md'), 'w') as f:
        f.write(f'| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ |\n|---|---|---|---|---|\n')
        for name, agg in summary.items():
            f.write(f'| {name} | {agg["psnr"][0]:.2f} ± {agg["psnr"][1]:.2f} '
                    f'| {agg["ssim"][0]:.3f} ± {agg["ssim"][1]:.3f} '
                    f'| {agg["lpips"][0]:.3f} ± {agg["lpips"][1]:.3f} '
                    f'| {agg["fvd"]:.1f} |\n')
        f.write(f'\n{len(segments)} segments from {len(files)} held-out BridgeData V2 episodes; '
                f'{ctx} context frames, {args.segment_length - ctx} predicted frames, 64x64, seed {args.seed}.\n')
    print(f'wrote {out_dir}/metrics.json, metrics.md, gifs/', flush=True)


if __name__ == '__main__':
    main()
