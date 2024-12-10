import h5py
import cv2
import imageio
import io
import hashlib
import numpy as np
import os
from tqdm import tqdm
import argparse

from robonet.metadata_helper import load_metadata


def load_camera_imgs(cam_index, file_pointer, file_metadata, target_dims, start_time=0, n_load=None):
    cam_group = file_pointer['env']['cam{}_video'.format(cam_index)]
    old_dims = file_metadata['frame_dim']
    length = file_metadata['img_T']
    encoding = file_metadata['img_encoding']
    image_format = file_metadata['image_format']

    if n_load is None:
        n_load = length

    old_height, old_width = old_dims

    images = np.zeros((n_load, old_height, old_width, 3), dtype=np.uint8)
    if encoding == 'mp4':
        buf = io.BytesIO(cam_group['frames'][:].tostring())
        img_buffer = [img for t, img in enumerate(imageio.get_reader(buf, format='mp4'))]
    elif encoding == 'jpg':
        img_buffer = [cv2.imdecode(cam_group['frame{}'.format(t)][:], cv2.IMREAD_COLOR)[:, :, ::-1]
                      for t in range(start_time, start_time + n_load)]
    else:
        raise ValueError("encoding not supported")

    for t, img in enumerate(img_buffer):
        images[t] = img

    if image_format == 'RGB':
        pass
    elif image_format == 'BGR':
        images = images[:, :, :, ::-1]
    else:
        raise NotImplementedError

    return images


def load_actions(file_pointer, meta_data):
    a_T, adim = meta_data['action_T'], meta_data['adim']
    if adim == 5:
        return file_pointer['policy']['actions'][:]
    elif adim == 4 and meta_data['primitives'] == 'autograsp':
        action_append, old_actions = np.zeros((a_T, 1)), file_pointer['policy']['actions'][:]
        next_state = file_pointer['env']['state'][:][1:, -1]

        high_val, low_val = meta_data['high_bound'][-1], meta_data['low_bound'][-1]
        midpoint = (high_val + low_val) / 2.0

        for t, s in enumerate(next_state):
            if s > midpoint:
                action_append[t, 0] = high_val
            else:
                action_append[t, 0] = low_val
        return np.concatenate((old_actions, action_append), axis=-1)
    elif adim < 4:
        pad = np.zeros((a_T, 5 - adim), dtype=np.float32)
        return np.concatenate((file_pointer['policy']['actions'][:], pad), axis=-1)
    elif adim > 5:
        return file_pointer['policy']['actions'][:][:, :5]


def load_data(f_name, file_metadata):
    assert os.path.exists(f_name) and os.path.isfile(f_name), "invalid f_name"
    with open(f_name, 'rb') as f:
        buf = f.read()
    assert hashlib.sha256(buf).hexdigest(
    ) == file_metadata['sha256'], "file hash doesn't match meta-data. maybe delete pkl and re-generate?"

    with h5py.File(io.BytesIO(buf)) as hf:
        start_time, n_states = 0, min([file_metadata['state_T'], file_metadata['img_T'], file_metadata['action_T'] + 1])
        assert n_states > 1, "must be more than one state in loaded tensor!"

        images, selected_cams = [], []
        images.append(load_camera_imgs(0, hf, file_metadata, None, start_time, n_states)[None])
        selected_cams.append(0)
        images = np.swapaxes(np.concatenate(images, 0), 0, 1)

        actions = load_actions(hf, file_metadata).astype(np.float32)[start_time:start_time + n_states - 1]

    return images, actions, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    file_list = os.listdir(args.hdf5_path)
    train_save_path = os.path.join(args.save_path, "train")
    test_save_path = os.path.join(args.save_path, "test")
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)

    test_file_list = []
    with open("datasets/robonet/robonet_testset_filenames.txt", 'r') as f:
        for line in f:
            test_file_list.append(line.strip())

    for index, file_name in tqdm(enumerate(file_list)):
        if ".pkl" in file_name:
            continue

        file_save_path = test_save_path if file_name in test_file_list else train_save_path
        save_name = os.path.join(file_save_path, file_name.split('.')[0] + '.npz')

        hdf5_file_name = os.path.join(args.hdf5_path, file_name)
        assert 'hdf5' in hdf5_file_name
        meta_data = load_metadata(hdf5_file_name)

        imgs, actions, _ = load_data(hdf5_file_name, meta_data.get_file_metadata(hdf5_file_name))

        img_shape = imgs.shape
        imgs = imgs.reshape((-1, img_shape[-3], img_shape[-2], img_shape[-1]))

        np.savez_compressed(save_name, **{'image': imgs, 'action': actions})
