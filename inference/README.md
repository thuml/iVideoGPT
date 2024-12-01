#  Inference Examples with Pre-trained iVideoGPTs

Action-free prediction:

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-64-act-free" --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data
```

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-64-act-free-medium" --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data
```

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-256-act-free" --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data --resolution 256 --repeat_times 1
```

Goal-conditioned prediction:

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-64-goal-cond" --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data --goal_conditioned
```

To try more samples, download the dataset from the [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/) and extract single episodes as follows:

```bash
python oxe_data_converter.py --dataset_name {dataset_name, e.g. bridge} --input_path {path to OXE} --output_path samples --max_num_episodes 10
```