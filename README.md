# iVideoGPT: Interactive VideoGPTs are Scalable World Models

[[Website]](https://thuml.github.io/iVideoGPT/) [[Paper]](https://arxiv.org/abs/2405.15223) [[Model]](https://huggingface.co/thuml/ivideogpt-oxe-64-act-free)

This repo provides official code and checkpoints for iVideoGPT, a generic and efficient world model architecture that has been pre-trained on millions of human and robotic manipulation trajectories. 

![architecture](assets/architecture.png)

## News

- 🚩 **2024.05.31**: Project website with video samples are released.
- 🚩 **2024.05.30**: Model pre-trained on Open X-Embodiment and inference code are released.
- 🚩 **2024.05.28**: The pre-trained model, inference code, project website, and a demo are coming soon in about one week!
- 🚩 **2024.05.27**: Our paper is released on [arXiv](https://arxiv.org/abs/2405.15223).

## Installation

```
conda create -n ivideogpt python==3.9
conda activate ivideogpt
pip install -r requirements.txt
```

## Models

At the moment we provide the following models:

| Model | Resolution | Action | Tokenizer Size | Transformer Size |
| ---- | ---- | ---- | ---- | ---- |
| [ivideogpt-oxe-64-act-free](https://huggingface.co/thuml/ivideogpt-oxe-64-act-free) | 64x64 | No |  114M   |  138M    |

If no network connection to Hugging Face, you can manually download from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/ef7d94c798504587a95e/).

## Inference Examples

### Action-free Video Prediction on Open X-Embodiment

```
python predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-64-act-free" --input_path samples/fractal_sample.npz --dataset_name fractal20220817_data
```

To try more samples, download the dataset from the [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/) and extract single episodes as follows:

```
python oxe_data_converter.py --dataset_name {dataset_name, e.g. bridge} --input_path {path to OXE} --output_path samples --max_num_episodes 10
```

## Showcases

![showcase](assets/showcase.png)

## Citation

If you find this project useful, please cite our paper as:

```
@article{wu2024ivideogpt,
    title={iVideoGPT: Interactive VideoGPTs are Scalable World Models}, 
    author={Jialong Wu and Shaofeng Yin and Ningya Feng and Xu He and Dong Li and Jianye Hao and Mingsheng Long},
    journal={arXiv preprint arXiv:2405.15223},
    year={2024},
}
```

## Contact

If you have any question, please contact wujialong0229@gmail.com.
