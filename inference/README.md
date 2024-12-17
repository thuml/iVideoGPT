#  Inference Examples with Pre-trained iVideoGPTs

## Action-free Prediction

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-64-act-free" --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data
```

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-64-act-free-medium" --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data
```

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-256-act-free" --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data --resolution 256 --repeat_times 1
```

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-bair-64-act-free" --input_path inference/samples/bair_sample.npz --dataset_name bair_robot_pushing --context_length 1
```

## Action-conditioned Prediction

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-robonet-64-act-cond" --input_path inference/samples/robonet_sample.npz --dataset_name tfds_robonet --context_length 2 --segment_length 12 --action_conditioned --action_dim 5
```

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-bair-64-act-cond" --input_path inference/samples/bair_sample.npz --dataset_name bair_robot_pushing --context_length 1 --segment_length 16 --action_conditioned --action_dim 4
```

<!-- ```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-robonet-256-act-cond" --input_path inference/samples/robonet_sample.npz --dataset_name tfds_robonet --context_length 2 --segment_length 12 --action_conditioned --action_dim 5 --resolution 256 --repeat_times 1
``` -->

## Goal-conditioned Prediction

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-64-goal-cond" --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data --goal_conditioned
```

## More Samples

To try more samples, download datasets from [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) and extract single episodes as follows:

```bash
python datasets/oxe_data_converter.py --dataset_name {dataset_name, e.g. bridge} --input_path {path to OXE} --output_path inference/samples --max_num_episodes 10
```