from typing import Optional, List, Tuple

import cv2
from torch import Tensor
import torch
import numpy as np
import torch.utils.data as data
import glob
import os
import yaml
import random
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import math
from .sthsth_dataloader import SomethingV2Dataset


def get_base_stepsize(dataset_name):
    stepsize = {
        'fractal20220817_data': 3,
        'kuka': 10,
        'bridge': 5,
        'taco_play': 15,
        'jaco_play': 10,
        'berkeley_cable_routing': 10,
        'roboturk': 10,
        'viola': 20,
        'toto': 30,
        'language_table': 10,
        'columbia_cairlab_pusht_real': 10,
        'stanford_kuka_multimodal_dataset_converted_externally_to_rlds': 20,
        'stanford_hydra_dataset_converted_externally_to_rlds': 10,
        'austin_buds_dataset_converted_externally_to_rlds': 20,
        'nyu_franka_play_dataset_converted_externally_to_rlds': 3,
        'maniskill_dataset_converted_externally_to_rlds': 20,
        'furniture_bench_dataset_converted_externally_to_rlds': 10,
        'ucsd_kitchen_dataset_converted_externally_to_rlds': 2,
        'ucsd_pick_and_place_dataset_converted_externally_to_rlds': 3,
        'austin_sailor_dataset_converted_externally_to_rlds': 20,
        'bc_z': 10,
        'utokyo_pr2_opening_fridge_converted_externally_to_rlds': 10,
        'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds': 10,
        'utokyo_xarm_pick_and_place_converted_externally_to_rlds': 10,
        'utokyo_xarm_bimanual_converted_externally_to_rlds': 10,
        'robo_net': 1,
        'kaist_nonprehensile_converted_externally_to_rlds': 10,
        'stanford_mask_vit_converted_externally_to_rlds': 1,  # no groundtruth value
        'dlr_sara_pour_converted_externally_to_rlds': 10,
        'dlr_sara_grid_clamp_converted_externally_to_rlds': 10,
        'dlr_edan_shared_control_converted_externally_to_rlds': 5,
        'asu_table_top_converted_externally_to_rlds': 12.5,
        'iamlab_cmu_pickup_insert_converted_externally_to_rlds': 20,
        'uiuc_d3field1': 1,
        'uiuc_d3field2': 1,
        'uiuc_d3field3': 1,
        'uiuc_d3field4': 1,
        'utaustin_mutex': 20,
        'berkeley_fanuc_manipulation': 10,
        'cmu_playing_with_food': 10,
        'cmu_play_fusion': 5,
        'cmu_stretch': 10,

        # downstream tasks
        'bair_robot_pushing': 1,
        'vp2_robodesk': 1,
        'vp2_robosuite': 1,
    }
    if dataset_name in stepsize:
        return stepsize[dataset_name]
    return 1


def get_display_key(dataset_name):
    key = {
        'taco_play': 'rgb_static',
        'roboturk': 'front_rgb',
        'viola': 'agentview_rgb',
        'berkeley_autolab_ur5': 'hand_image',
        'language_table': 'rgb',
        'berkeley_mvp_converted_externally_to_rlds': 'hand_image',
        'berkeley_rpt_converted_externally_to_rlds': 'hand_image',
        'stanford_robocook_converted_externally_to_rlds1': 'image_1',
        'stanford_robocook_converted_externally_to_rlds2': 'image_2',
        'stanford_robocook_converted_externally_to_rlds3': 'image_3',
        'stanford_robocook_converted_externally_to_rlds4': 'image_4',
        'uiuc_d3field1': 'image_1',
        'uiuc_d3field2': 'image_2',
        'uiuc_d3field3': 'image_3',
        'uiuc_d3field4': 'image_4',

        # downstream tasks
        'bair_robot_pushing': 'aux1_image',
        'vp2_robodesk': 'image',
        'vp2_robosuite': 'image',
    }
    if dataset_name in key:
        return key[dataset_name]
    return 'image'


class SimpleRoboticDatasetv2(data.Dataset):
    def __init__(
        self, parent_dir, dataset_name,
        # segment
        random_selection,
        random_shuffle,
        goal_conditioned,
        segment_length,
        context_length,
        stepsize,
        segment_horizon,
        # augmentation
        random_resized_crop_scale=None,
        random_resized_crop_ratio=None,
        brightness=None,
        contrast=None,
        saturation=None,
        hue=None,
        no_aug=False,
        # split
        train=True,
        maxsize=None,
        image_size=256,
        # action
        load_action=False,
    ):
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.train = train

        self.segment_length = segment_length
        self.context_length = context_length
        self.random_selection = random_selection
        self.random_shuffle = random_shuffle
        self.goal_conditioned = goal_conditioned
        self.segment_horizon = segment_horizon or segment_length
        self.stepsize = stepsize

        self.random_resized_crop_scale = random_resized_crop_scale
        self.random_resized_crop_ratio = random_resized_crop_ratio
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.no_aug = no_aug

        self.load_action = load_action

        if dataset_name == 'bair_robot_pushing':
            if train:
                parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['bair_train_dataset']
            else:
                parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['bair_test_dataset']
            self.filenames = glob.glob(os.path.join(parent_dir, '*.npz'))
            self.filenames.sort()
        elif dataset_name == 'vp2_robodesk':
            parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robodesk_dataset']
            if train:
                self.filenames = glob.glob(os.path.join(parent_dir, '*', 'train*', '*.npz'))
            else:
                self.filenames = glob.glob(os.path.join(parent_dir, '*', 'validation*', '*.npz'))
            self.filenames.sort()
        elif dataset_name == 'vp2_robosuite':
            parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robosuite_dataset']
            if train:
                self.filenames = glob.glob(os.path.join(parent_dir, 'train', '*.npz'))
            else:
                self.filenames = glob.glob(os.path.join(parent_dir, 'validation', '*.npz'))
            self.filenames.sort()
        elif dataset_name == 'tfds_robonet':
            if train:
                parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robonet_train_dataset']
            else:
                parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robonet_test_dataset']
            self.filenames = glob.glob(os.path.join(parent_dir, '*.npz'))
            self.filenames.sort()
        else:
            self.filenames = glob.glob(os.path.join(parent_dir, dataset_name, '*.npz'))
            self.filenames.sort()
            if train:
                self.filenames = [x for i, x in enumerate(self.filenames) if i % 100 != 0]
            else:
                self.filenames = [x for i, x in enumerate(self.filenames) if i % 100 == 0]
            if dataset_name == 'robo_net':
                with open('datasets/robonet/oxe_robonet_testset_filenames.txt', 'r') as f:
                    robonet_testset = [line.strip().split()[1] for line in f.readlines()]
                original_size = len(self.filenames)
                self.filenames = [x for x in self.filenames if os.path.basename(x) not in robonet_testset]
                print(
                    f"[SimpleRoboticDataset] Filter out {original_size - len(self.filenames)} {'train' if train else 'test'} episodes in {dataset_name}")
        if maxsize is not None:
            self.size = maxsize
            state = random.getstate()
            random.seed(0)
            self.filenames = random.choices(self.filenames, k=maxsize)   # !BUG: this is with replacement!
            print(
                f"[SimpleRoboticDataset] Randomly select {maxsize} {'train' if train else 'test'} episodes in {dataset_name}: ", self.filenames[:5])
            random.setstate(state)
        else:
            self.size = len(self.filenames)
        self.display_key = get_display_key(dataset_name)
        if self.size == 0:
            raise ValueError(
                f"[SimpleRoboticDataset] No {'train' if train else 'test'} episodes found in {dataset_name}")
        print(
            f"[SimpleRoboticDataset] Find {self.size} {'train' if train else 'test'} episodes with stepsize {stepsize} in {dataset_name}")

    def set_horizon(self, horizon):
        self.segment_horizon = horizon

    @staticmethod
    def get_crop_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        _, height, width = F.get_dimensions(img)
        # area = height * width
        area = min(height, width) ** 2

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    @staticmethod
    def get_jittor_params(
        brightness: Optional[List[float]],
        contrast: Optional[List[float]],
        saturation: Optional[List[float]],
        hue: Optional[List[float]],
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def get_segment(self, episode, action=None):
        if self.goal_conditioned:
            segment_length = self.segment_length - 1
            # shrink stepsize if episode too short
            if self.stepsize * segment_length > len(episode):
                stepsize = max(1, len(episode) // segment_length)
            else:
                stepsize = self.stepsize

            start = np.random.randint(max(len(episode) - stepsize * segment_length + 1, 1))
            images = [episode[min(start + stepsize * i, len(episode) - 1)] for i in range(segment_length)]
            images = images[-1:] + images  # last frame as goal
            if action is not None:
                raise NotImplementedError
            else:
                actions = None
        elif self.random_shuffle:
            # shrink stepsize if episode too short
            if self.stepsize * self.segment_horizon > len(episode):
                stepsize = max(1, len(episode) // self.segment_horizon)
            else:
                stepsize = self.stepsize

            start = np.random.randint(max(len(episode) - stepsize * self.segment_horizon + 1, 1))
            idx = np.random.choice(self.segment_horizon, self.segment_length, replace=False)
            images = [episode[min(start + stepsize * i, len(episode) - 1)] for i in idx]
            if action is not None:
                raise NotImplementedError
            else:
                actions = None
        elif self.random_selection:
            # shrink stepsize if episode too short
            if self.stepsize * self.segment_horizon > len(episode):
                stepsize = max(1, len(episode) // self.segment_horizon)
            else:
                stepsize = self.stepsize

            start = np.random.randint(max(len(episode) - stepsize * self.segment_horizon + 1, 1))
            all_images = [step for step in episode[start: start + stepsize * self.segment_horizon]]
            context_images = all_images[:stepsize * self.context_length:stepsize]
            after_images = all_images[stepsize * self.context_length:]
            idx = np.random.choice(len(after_images), min(
                len(after_images), self.segment_length - self.context_length), replace=False)
            idx = np.sort(idx)
            after_images = [after_images[i] for i in idx]
            images = context_images + after_images
            if action is not None:
                all_actions = [action for action in action[start: start + stepsize * self.segment_horizon]]
                context_actions = all_actions[:stepsize * self.context_length:stepsize]
                after_actions = all_actions[stepsize * self.context_length:]
                after_actions = [after_actions[i] for i in idx]
                actions = context_actions + after_actions
            else:
                actions = None
        else:
            # shrink stepsize if episode too short
            if self.stepsize * self.segment_length > len(episode):
                stepsize = max(1, len(episode) // self.segment_length)
            else:
                stepsize = self.stepsize

            # use EvalDataset for downstream task fixed evaluation
            start = np.random.randint(max(len(episode) - stepsize * self.segment_length + 1, 1))
            images = [step for step in episode[start: start + stepsize * self.segment_length: stepsize]]
            if action is not None:
                actions = [action for action in action[start: start + stepsize * self.segment_length: stepsize]]
            else:
                actions = None

        # if the episode is too short, repeat the last image
        while len(images) < self.segment_length:
            images.append(images[-1])
            if action is not None:
                actions.append(actions[-1])
        return images, actions

    def data_augmentation(self, images):
        i, j, h, w = self.get_crop_params(images, self.random_resized_crop_scale, self.random_resized_crop_ratio)

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_jittor_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        new_images = []
        transform = transforms.ToPILImage()
        tensor = transforms.ToTensor()
        for image in images:
            image = F.resized_crop(image, i, j, h, w, [self.image_size, self.image_size])
            image = transform(image / 255)
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    image = F.adjust_brightness(image, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    image = F.adjust_contrast(image, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    image = F.adjust_saturation(image, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    image = F.adjust_hue(image, hue_factor)
            # image = F.resized_crop(image, i, j, h, w, [self.image_size, self.image_size])
            image = tensor(image)
            new_images.append(image)
        return torch.stack(new_images)

    def centercrop(self, images):
        images = images / 255
        if self.dataset_name == 'tfds_robonet':
            images = F.center_crop(images, min(images.shape[2:]))
        return F.resize(images, [self.image_size, self.image_size])

    def __getitem__(self, item):
        id = np.random.randint(self.size)
        episode = np.load(self.filenames[id])[self.display_key]
        action = np.load(self.filenames[id])['action'] if self.load_action else None
        if self.dataset_name == 'tfds_robonet' and action is not None:
            new_row = np.array([0, 0, 0, 0, 0]).reshape(1, -1)
            action = np.append(action, new_row, axis=0)
        images, actions = self.get_segment(episode, action)
        images = torch.Tensor(np.array(images)).permute(0, 3, 1, 2)  # T, H, W, C -> T, C, H, W
        if self.no_aug:
            images = self.centercrop(images)
        else:
            images = self.data_augmentation(images)

        if self.load_action:
            actions = torch.Tensor(np.array(actions))
            return images, actions
        else:
            return images

    def __len__(self):
        return self.size * 10000000  # infinity


class MixRoboticDatasetv2(data.Dataset):
    def __init__(self, parent_dir, datasets, stepsize=1, sthsth_root_path=None, **dataset_args):

        # datasets: [(name, mix), ...]
        dataset_names = []
        dataset_mix = []
        self.dataset_sizes = []
        for dataset in datasets:
            name, mix = dataset
            dataset_names.append(name)
            dataset_mix.append(mix)
        frac_step_size = 3
        self.datasets = [
            SimpleRoboticDatasetv2(
                parent_dir, dataset_name,
                # get the stepsize for each dataset, stepsize should be at least 1
                stepsize=max(round(stepsize * get_base_stepsize(dataset_name) / frac_step_size), 1),
                **dataset_args
            ) if dataset_name != 'sthsth' else
            SomethingV2Dataset(
                sthsth_root_path,
                stepsize=1,
                **dataset_args
            )
            for dataset_name in dataset_names
        ]

        for dataset in self.datasets:
            self.dataset_sizes.append(dataset.size)
        self.sample_weights = np.array(dataset_mix)
        self.num_datasets = len(datasets)

    def __getitem__(self, item):
        dataset_index = np.random.choice(self.num_datasets, p=self.sample_weights / self.sample_weights.sum())
        # item is not used
        return self.datasets[dataset_index][0]

    def __len__(self):
        return sum(self.dataset_sizes) * 100000000


class SimpleRoboticDataLoaderv2(data.DataLoader):
    def __init__(self, parent_dir, datasets, batch_size=2, num_workers=1, **dataset_args):
        self.dataset = MixRoboticDatasetv2(parent_dir, datasets, **dataset_args)
        super().__init__(self.dataset, batch_size=batch_size, num_workers=num_workers)


class EvalDataset(data.Dataset):
    def __init__(
        self, dataset_name,
        segment_length,
        image_size=256,
        load_action=False,
    ):
        self.image_size = image_size
        self.dataset_name = dataset_name

        self.segment_length = segment_length
        self.load_action = load_action

        if dataset_name == 'bair_robot_pushing':
            parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['bair_test_dataset']
            self.filenames = glob.glob(os.path.join(parent_dir, '*.npz'))
            self.filenames.sort()
        elif dataset_name == 'tfds_robonet':
            parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robonet_test_dataset']
            self.filenames = glob.glob(os.path.join(parent_dir, '*.npz'))
            self.filenames.sort()
        elif dataset_name == 'vp2_robodesk':
            parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robodesk_dataset']
            self.filenames = glob.glob(os.path.join(parent_dir, '*', 'validation*', '*.npz'))
            self.filenames.sort()
            self.filenames.sort()
        elif dataset_name == 'vp2_robosuite':
            parent_dir = yaml.load(open('DATASET.yaml'), Loader=yaml.FullLoader)['robosuite_dataset']
            self.filenames = glob.glob(os.path.join(parent_dir, 'validation', '*.npz'))
            self.filenames.sort()
        else:
            raise NotImplementedError

        self.size = len(self.filenames)
        self.display_key = get_display_key(dataset_name)
        if self.size == 0:
            raise ValueError(f"[EvalDataset] No test episodes found in {dataset_name}")
        print(f"[EvalDataset] Find {self.size} test episodes in {dataset_name}")

    def data_augmentation(self, images):
        images = images / 255
        if self.dataset_name == 'tfds_robonet':
            images = F.center_crop(images, min(images.shape[2:]))
        return F.resize(images, [self.image_size, self.image_size])

    def get_segment(self, episode, action=None):
        if 'vp2' in self.dataset_name:
            start = np.random.randint(max(len(episode) - self.segment_length + 1, 1))
        else:
            start = 0  # for test on downstream video prediction tasks
        images = [step for step in episode[start: start + self.segment_length]]
        if action is not None:
            actions = [action for action in action[start: start + self.segment_length]]
        else:
            actions = None
        # if the episode is too short, repeat the last image
        while len(images) < self.segment_length:
            images.append(images[-1])
            if action is not None:
                actions.append(actions[-1])
        return images, actions

    def __getitem__(self, item):
        episode = np.load(self.filenames[item])[self.display_key]
        action = np.load(self.filenames[item])['action'] if self.load_action else None
        if self.dataset_name == 'tfds_robonet' and action is not None:
            new_row = np.array([0, 0, 0, 0, 0]).reshape(1, -1)
            action = np.append(action, new_row, axis=0)
        images, actions = self.get_segment(episode, action)
        images = torch.Tensor(np.array(images)).permute(0, 3, 1, 2)  # T, H, W, C -> T, C, H, W
        images = self.data_augmentation(images)

        if self.load_action:
            actions = torch.Tensor(np.array(actions))
            return images, actions
        else:
            return images

    def __len__(self):
        return self.size


class EvalDataLoader(data.DataLoader):
    def __init__(self, dataset_name, segment_length, image_size, batch_size=2, num_workers=1, load_action=False, **dummy_args):
        self.dataset = EvalDataset(dataset_name, segment_length, image_size, load_action)
        super().__init__(self.dataset, batch_size=batch_size, num_workers=num_workers)
