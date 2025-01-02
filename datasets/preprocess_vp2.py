import argparse
import h5py
import os
import numpy as np
from tqdm import tqdm


def load_and_convert(file_path, save_path, mode='train'):
    assert mode in ['train', 'valid']
    hdf5_file = h5py.File(file_path, 'r', swmr=False, libver='latest')

    if mode == 'train':
        mode_name = "train"
        demos = [elem.decode("utf-8") for elem in np.array(hdf5_file["mask/train"][:])]
    else:
        mode_name = "validation"
        demos = [elem.decode("utf-8") for elem in np.array(hdf5_file["mask/valid"][:])]

    print(f"processing file {file_path}, mode={mode}")
    for demo in tqdm(demos):
        demo_name = demo[:5] + demo[5:].zfill(5)
        if os.path.exists(os.path.join(save_path, f'{mode_name}_eps_{demo_name}.npz')):
            continue
        if "robodesk" in save_path:
            obs = hdf5_file[f"data/{demo}/obs/camera_image"][()]
        else:
            assert "robosuite" in save_path
            obs = hdf5_file[f"data/{demo}/obs/agentview_shift_2_image"][()]
        actions = hdf5_file[f"data/{demo}/actions"][()]
        np.savez_compressed(os.path.join(save_path, f'{mode_name}_eps_{demo_name}.npz'),
                            **{'image': obs, 'action': actions})


def process_files(file_dir_path, save_dir_path, mode):
    files_list = os.listdir(file_dir_path)
    for file_or_dir in files_list:
        file_or_dir_path = os.path.join(file_dir_path, file_or_dir)
        if os.path.isdir(file_or_dir_path):
            new_save_dir_path = os.path.join(save_dir_path, file_or_dir)
            if not os.path.exists(new_save_dir_path):
                os.mkdir(new_save_dir_path)
            process_files(file_or_dir_path, new_save_dir_path, mode=mode)
        else:
            if "robodesk" in file_or_dir_path:
                if mode == 'train':
                    mode_name = "train"
                else:
                    mode_name = "validation"
                if "noise_0.1" in file_or_dir_path:
                    new_save_dir_path = os.path.join(save_dir_path, f"{mode_name}_noise1")
                elif "noise_0.2" in file_or_dir_path:
                    new_save_dir_path = os.path.join(save_dir_path, f"{mode_name}_noise2")
                else:
                    assert False
                if not os.path.exists(new_save_dir_path):
                    os.mkdir(new_save_dir_path)
                load_and_convert(file_or_dir_path, new_save_dir_path, mode=mode)
            else:
                if mode == 'train':
                    mode_name = "train"
                else:
                    mode_name = "validation"
                new_save_dir_path = os.path.join(save_dir_path, mode_name)
                if not os.path.exists(new_save_dir_path):
                    os.mkdir(new_save_dir_path)
                load_and_convert(file_or_dir_path, new_save_dir_path, mode=mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    assert os.path.exists(args.dir_path)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    process_files(args.dir_path, args.save_path, mode="train")
    process_files(args.dir_path, args.save_path, mode="valid")
