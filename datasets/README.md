# Dataset Preparation

## Open X-Embodiment

Download datasets from [Open X-Embodiment](https://robotics-transformer-x.github.io/) and extract single episodes as `.npz` files:

```bash
python datasets/oxe_data_converter.py --dataset_name {dataset name, e.g. bridge} --input_path {path to downloaded OXE} --output_path {path to stored npz}
```

For replicating our pre-training on OXE, you need to extract all datasets listed as `OXE_SELECT` in `ivideogpt/data/dataset_mixes.py`.

## Something-Something V2

Follow [ContextWM](https://github.com/thuml/ContextWM?tab=readme-ov-file#datasets) to prepare the Something-Something-V2 dataset. 

You should include `train_video_folder.txt` and `val_video_folder.txt` in the directory `datasets/somethingv2`.

## BAIR Robot Pushing

Download the [dataset](http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar) and preprocess with the following script:

```bash
wget http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar -P .
tar -xvf ./bair_robot_pushing_dataset_v0.tar -C .

python datasets/preprocess_bair.py --input_path bair_robot_pushing_dataset_v0/softmotion30_44k --save_path bair_preprocessed
```

Then modify the saved paths (e.g. `bair_preprocessed/train` and `bair_preprocessed/test`) in `DATASET.yaml`.

<!-- ## RoboNet -->