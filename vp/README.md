# Visual Planning

## Installation

Clone the repository of [VP2](https://github.com/s-tian/vp2) and install the required dependencies as well as the data containing task instance specifications following the instructions of VP2. 

We have made sure that there are no dependency conflicts between ivideogpt and vp2, so if you encounter trouble in the environments, you can follow these: 

1.  Create a new virtual environment.
2.  Install the iVideoGPT dependencies as described in the Installation section of `ivideopgt/README.md`. 
3.  Copy the provided `vp2_requirements.txt` file into the downloaded VP2 folder.
4.  Run `pip install -r vp2_requirements.txt`.

## Setting Up the iVideoGPT Interface

Follow the steps below to integrate the iVideoGPT interface into VP2:

1. Update the path in `ivideogpt_interface.py`: Go to line 12, where you will see `sys.path.append("/dev/null/ivideogpt")`.Replace `/dev/null/ivideogpt` with the absolute path to your local iVideoGPT folder. This step ensures relative paths are imported correctly.
2. Place `ivideogpt_interface.py` in the correct location: Copy it to `vp2/vp2/models/ivideogpt_interface.py`. You should see `torch_fitvid_interface.py` in the same directory.
3. Place `ivideogpt.yaml` in the correct location: Copy it to `vp2/vp2/scripts/configs/model/ivideogpt.yaml`. You should see `fitvid.yaml` in the same directory.

## Run

To test your trained iVideoGPT models on VP2:

1. In `ivideogpt.yaml`, specify the following: `config_name`, `pretrained_vqgan_name_or_path` and `pretrained_transformer_path`
2. Refer to `script.sh` for example usage instructions (Note: The working directory for these instructions is assumed to be `vp2/vp2`).
