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

## Goal-conditioned Prediction

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-64-goal-cond" --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data --goal_conditioned
```

## More Samples

To try more samples, download datasets from [Open X-Embodiment](https://robotics-transformer-x.github.io/) and extract single episodes as follows:

```bash
python datasets/oxe_data_converter.py --dataset_name {dataset_name, e.g. bridge} --input_path {path to OXE} --output_path inference/samples --max_num_episodes 10
```