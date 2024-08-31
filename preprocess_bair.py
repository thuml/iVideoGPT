import os
import numpy as np
import tensorflow as tf
from PIL import Image
import glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--save_gif', default=False, action='store_true')
args = parser.parse_args()

if args.save_gif:
    os.makedirs(os.path.join(args.save_path, 'gif'), exist_ok=True)

for split in ["train", "test"]:
    os.makedirs(os.path.join(args.save_path, split), exist_ok=True)

    data_dir = os.path.join(args.input_path, split)
    filenames_ = glob.glob(os.path.join(data_dir, '*.tfrecords'))
    assert len(filenames_) > 0, f"No tfrecords files found in {data_dir}"
    filenames = []

    for filename in filenames_:  # rename them, eg. change 0 to 000000. For sort.
        single_filename = filename.split('/')[-1].split('.')[0].strip()
        start_num = single_filename.split('_to_')[0].split('traj_')[-1]
        end_num = single_filename.split('_to_')[-1]
        start_num = start_num.rjust(6, '0')
        end_num = end_num.rjust(6, '0')
        new_filename = os.path.join(data_dir, 'traj_' + str(start_num) + '_to_' + str(end_num) + '.tfrecords')
        filenames.append(new_filename)

    filenames = sorted(filenames)
    for f in tqdm(filenames):
        single_filename = f.split('/')[-1].split('.')[0].strip()
        start_num = str(int(single_filename.split('_to_')[0].split('traj_')[-1]))
        start_num_int = int(start_num)
        file_index = start_num_int - 1

        end_num = str(int(single_filename.split('_to_')[-1]))
        f = os.path.join(data_dir, 'traj_' + str(start_num) + '_to_' + str(end_num) + '.tfrecords')
        for serialized_example in tf.compat.v1.io.tf_record_iterator(f):
            file_index += 1
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            main_image_np_list = []
            aux1_image_np_list = []
            action_np_list = []
            image_list = []
            for i in range(30):
                image_aux1_name = f'{i}/image_aux1/encoded'
                image_main_name = f'{i}/image_main/encoded'
                action_name = f'{i}/action'
                image_main_byte_str = example.features.feature[image_main_name].bytes_list.value[0]
                image_aux1_byte_str = example.features.feature[image_aux1_name].bytes_list.value[0]
                action_list = [example.features.feature[action_name].float_list.value[action_index]
                               for action_index in range(0, 4)]
                main_img = Image.frombytes('RGB', (64, 64), image_main_byte_str)
                image_list.append(main_img)
                main_image_np = np.array(main_img.getdata()).reshape((main_img.size[1], main_img.size[0], 3))
                aux1_img = Image.frombytes('RGB', (64, 64), image_aux1_byte_str)
                aux1_image_np = np.array(aux1_img.getdata()).reshape((aux1_img.size[1], aux1_img.size[0], 3))
                action_np = np.array(action_list)

                main_image_np_list.append(main_image_np)
                aux1_image_np_list.append(aux1_image_np)
                action_np_list.append(action_np)

            main_images = np.stack(main_image_np_list)
            aux1_images = np.stack(aux1_image_np_list)
            actions = np.stack(action_np_list)

            save_file_path = os.path.join(args.save_path, split, f"traj_{str(file_index).zfill(5)}.npz")
            np.savez_compressed(save_file_path, **{'image': main_images, 'action': actions, 'aux1_image': aux1_images})

            if args.save_gif:
                images = [Image.fromarray(np.uint8(image)) for image in aux1_images]
                gif_save_path = os.path.join(args.save_path, "gif", f"traj_{str(file_index).zfill(5)}.gif")
                images[0].save(gif_save_path, save_all=True, append_images=images[1:], duration=50, loop=0)
