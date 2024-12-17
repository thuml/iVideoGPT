import numpy as np
import torch
import torchvision.transforms.functional as F


class NPZParser:

    def __init__(self, segment_length, image_size=64):
        self.segment_length = segment_length
        self.image_size = image_size

    def preprocess(self, images):
        images = images / 255
        # images = F.center_crop(images, min(images.shape[2:]))
        images = F.resize(images, [self.image_size, self.image_size])
        return images

    def get_segment(self, episode, actions, stepsize=1):
        # shrink stepsize if episode too short
        if stepsize * self.segment_length > len(episode):
            stepsize = max(1, len(episode) // self.segment_length)

        start = np.random.randint(max(len(episode) - stepsize * self.segment_length + 1, 1))
        images = [step for step in episode[start: start + stepsize * self.segment_length: stepsize]]
        if actions is not None:
            actions = [action for action in actions[start: start + stepsize * self.segment_length: stepsize]]
        return images, actions

    def get_stepsize(self, dataset_name):
        return max(round(BASE_STEPSIZE.get(dataset_name, 1) / BASE_STEPSIZE['fractal20220817_data']), 1)

    def parse(self, npz_file, dataset_name, load_action=False):
        images = np.load(npz_file)[DISPLAY_KEY.get(dataset_name, 'image')]
        actions = np.load(npz_file)['action'] if load_action else None
        images, actions = self.get_segment(images, actions, self.get_stepsize(dataset_name))
        images = torch.Tensor(np.array(images)).permute(0, 3, 1, 2)  # T, H, W, C -> T, C, H, W
        images = self.preprocess(images)
        actions = torch.Tensor(np.array(actions)) if actions is not None else None
        return images, actions


BASE_STEPSIZE = {
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

    'bair_robot_pushing': 1,
    'tfds_robonet': 1,
}


DISPLAY_KEY = {
    'taco_play': 'rgb_static',
    'roboturk': 'front_rgb',
    'viola': 'agentview_rgb',
    'language_table': 'rgb',
    'stanford_robocook_converted_externally_to_rlds1': 'image_1',
    'stanford_robocook_converted_externally_to_rlds2': 'image_2',
    'stanford_robocook_converted_externally_to_rlds3': 'image_3',
    'stanford_robocook_converted_externally_to_rlds4': 'image_4',
    'uiuc_d3field1': 'image_1',
    'uiuc_d3field2': 'image_2',
    'uiuc_d3field3': 'image_3',
    'uiuc_d3field4': 'image_4',
    
    'bair_robot_pushing': 'aux1_image',
    'tfds_robonet': 'image',
}
