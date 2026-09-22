"""Extract BridgeData V2 episodes from locally downloaded TFDS shards to .npz files.

Reads tfrecord shards directly (works with a partial download of the 1024-shard
dataset), deserializes with the TFDS feature spec, and saves one .npz per episode
with keys: image (T, 256, 256, 3), action (T, 7).

Usage:
  python datasets/bridge_v2_extract.py --builder_dir data/bridge_dataset/1.0.0 \
      --split train --output_path data/bridge_npz/train --max_num_episodes 100
"""
import argparse
import glob
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

tf.config.set_visible_devices([], 'GPU')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--builder_dir', type=str, default='data/bridge_dataset/1.0.0')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--output_path', type=str, default='data/bridge_npz/train')
    parser.add_argument('--max_num_episodes', type=int, default=None)
    parser.add_argument('--image_key', type=str, default='image_0')
    args = parser.parse_args()

    features = tfds.builder_from_directory(builder_dir=args.builder_dir).info.features
    shards = sorted(glob.glob(os.path.join(args.builder_dir, f'bridge_dataset-{args.split}.tfrecord-*')))
    assert shards, f'no {args.split} shards found in {args.builder_dir}'
    print(f'{len(shards)} local {args.split} shard(s)')

    os.makedirs(args.output_path, exist_ok=True)
    num_episodes = 0
    skipped = 0
    for shard in shards:
        for raw in tqdm(tf.data.TFRecordDataset(shard), desc=os.path.basename(shard)):
            episode = features.deserialize_example(raw)
            out_file = os.path.join(args.output_path, f'{args.split}_eps_{num_episodes:08d}.npz')
            frames, actions = [], []
            for step in episode['steps']:
                frames.append(step['observation'][args.image_key].numpy())
                actions.append(step['action'].numpy())
            frames = np.array(frames)
            # some episodes lack the chosen camera (all-black frames) — skip them
            if frames.max() == 0:
                skipped += 1
                continue
            np.savez_compressed(out_file, image=frames, action=np.array(actions))
            num_episodes += 1
            if args.max_num_episodes and num_episodes >= args.max_num_episodes:
                print(f'saved {num_episodes} episodes to {args.output_path} (skipped {skipped} without {args.image_key})')
                return
    print(f'saved {num_episodes} episodes to {args.output_path} (skipped {skipped} without {args.image_key})')


if __name__ == '__main__':
    main()
