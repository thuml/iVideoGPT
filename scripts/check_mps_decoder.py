"""CPU-vs-MPS consistency check for CompressiveVQModel.detokenize (issue #29).

Tokenizes a 64x64 sample on CPU, detokenizes the same ids on CPU and on MPS, and reports
the PSNR between the two reconstructions plus the first leaf module whose output diverges.

    PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/check_mps_decoder.py
"""
import numpy as np
import torch
import torch.nn.functional as F

from ivideogpt.vq_model import CompressiveVQModel

MODEL = "pretrained_models/ivideogpt-oxe-64-act-free"
SAMPLE = "inference/samples/fractal_sample.npz"


def psnr(a, b):
    return float(10 * torch.log10(1.0 / ((a - b) ** 2).mean().clamp_min(1e-12)))


def capture(model, device, tokens):
    recs, hooks = [], []

    def mk(name):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o):
                recs.append((name, o.detach().float().cpu()))
        return hook

    for name, m in model.named_modules():
        if name and not list(m.children()):
            hooks.append(m.register_forward_hook(mk(name)))
    with torch.no_grad():
        out = model.detokenize(tokens.to(device), 2).clamp(0, 1).cpu()
    for h in hooks:
        h.remove()
    return recs, out


def main():
    assert torch.backends.mps.is_available(), "needs an MPS device"
    tok = CompressiveVQModel.from_pretrained(MODEL, subfolder="tokenizer", low_cpu_mem_usage=False).eval()
    x = torch.from_numpy(np.load(SAMPLE)["image"]).float()
    if x.max() > 1.5:
        x = x / 255.0
    if x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    x = F.interpolate(x[:16], size=(64, 64), mode="bilinear", align_corners=False, antialias=True)
    x = x.view(1, -1, 3, 64, 64).contiguous()

    with torch.no_grad():
        ids, _ = tok.tokenize(x, 2)
    recs_cpu, out_cpu = capture(tok, "cpu", ids)
    recs_mps, out_mps = capture(tok.to("mps"), "mps", ids)
    print(f"torch {torch.__version__}: CPU recon PSNR vs input {psnr(out_cpu, x):.2f} dB; "
          f"MPS vs CPU {psnr(out_cpu, out_mps):.2f} dB, max|diff| {(out_cpu - out_mps).abs().max():.4f}")
    for (name, a), (_, b) in zip(recs_cpu, recs_mps):
        rel = ((a - b).abs().max() / (a.abs().max() + 1e-8)).item()
        if rel > 1e-3:
            print(f"first divergence: {name} (max rel diff {rel:.2e}, shape {tuple(a.shape)})")
            return 1
    print("OK: MPS decoder matches CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
