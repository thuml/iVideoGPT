"""Inspect extracted BridgeData V2 episodes: stats, frame montage, raw episode GIFs."""
import glob
import json
import os

import imageio
import numpy as np
from PIL import Image, ImageDraw

NPZ_DIR = 'data/bridge_npz/val'
OUT_DIR = 'outputs/data_inspection'
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(NPZ_DIR, '*.npz')))
print(f'{len(files)} episodes in {NPZ_DIR}\n')

lengths, action_mins, action_maxs, action_means = [], [], [], []
for f in files:
    d = np.load(f)
    img, act = d['image'], d['action']
    lengths.append(len(img))
    action_mins.append(act.min(axis=0))
    action_maxs.append(act.max(axis=0))
    action_means.append(act.mean(axis=0))

lengths = np.array(lengths)
print(f'episode length: min {lengths.min()}, median {int(np.median(lengths))}, max {lengths.max()}')
d0 = np.load(files[0])
print(f'frame shape: {d0["image"].shape[1:]}, dtype {d0["image"].dtype}')
print(f'action shape per step: {d0["action"].shape[1:]}, dtype {d0["action"].dtype}')

act_min = np.min(action_mins, axis=0)
act_max = np.max(action_maxs, axis=0)
dim_names = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
print('\naction ranges across episodes (7-dim: xyz delta, rpy delta, gripper):')
for i, n in enumerate(dim_names):
    print(f'  {n:8s} [{act_min[i]:+.4f}, {act_max[i]:+.4f}]')

# montage: 6 episodes x 5 timesteps
N_EPS, N_T = 6, 5
thumb = 128
pad = 4
label_h = 18
W = N_T * (thumb + pad) + pad
H = N_EPS * (thumb + pad + label_h) + pad
canvas = Image.new('RGB', (W, H), (24, 24, 24))
draw = ImageDraw.Draw(canvas)
for r, f in enumerate(files[:N_EPS]):
    img = np.load(f)['image']
    idxs = np.linspace(0, len(img) - 1, N_T).astype(int)
    y = pad + r * (thumb + pad + label_h)
    draw.text((pad, y + 2), f'{os.path.basename(f)}  (T={len(img)})', fill=(200, 200, 200))
    for c, t in enumerate(idxs):
        im = Image.fromarray(img[t]).resize((thumb, thumb))
        canvas.paste(im, (pad + c * (thumb + pad), y + label_h))
canvas.save(os.path.join(OUT_DIR, 'bridge_montage.png'))
print(f'\nmontage saved: {OUT_DIR}/bridge_montage.png (rows = episodes, cols = time)')

# raw episode GIFs at full res
for f in files[:3]:
    img = np.load(f)['image']
    name = os.path.basename(f).replace('.npz', '')
    imageio.mimsave(os.path.join(OUT_DIR, f'raw_{name}.gif'), list(img), fps=5, loop=0)
print(f'raw GIFs saved for 3 episodes')
