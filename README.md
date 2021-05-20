# background_foreground_segmentation

Structure allows for standalone python usage and usage within ROS.

# Installation
The software is organized as a hybrid ROS and python workspace. Localisation experiments and online learning are running in the ROS workspace. Segmentation training and evaluation are running in the python-only workspace. Please follow the respective instructions below to set up the different workspaces.

## Installing the ROS workspace
First, setup a catkin workspace on ROS melodic. We usually follow [this guide](https://github.com/ethz-asl/maplab/wiki/Installation-Ubuntu#create-a-catkin-workspace) for our workspaces.

Use `wstool` or some other ROS dependency manager to install all packages from `dependencies.rosinstall`.

## Installing the python workspace
### Create virtualenv
Use your favourite way to create a python environment. We recommend one of these:

```bash
mkvirtualenv background_foreground_segmentation --python=$(which python3)
```

```bash
python3 -m venv py3_venv
```

### Install dependencies
Letting `$BFSEG_ROOT` be the folder of this repo (i.e., where this README is located), assuming the virtualenv created above is always sourced:
- Install required dependencies:
  ```bash
  cd $BFSEG_ROOT
  pip install -r requirements.txt
  ```
- Install Python package in this repo:
  ```bash
  cd $BFSEG_ROOT/src
  pip install -e .
  ```

### Load Datasets
For training and evaluation, some datasets are required. We use [TFDS](https://www.tensorflow.org/datasets) to automatically download and extract these datasets. This will require around 50GB and can take a couple of hours to prepare.

```bash
cd $BFSEG_ROOT
python data_setup.py
```

### Configure Settings
create a file `src/bfseg/settings.py` with the following content:
```python
TMPDIR = '/tmp'  # a temporary storage folder
EXPERIMENT_STORAGE_FOLDER = '<insert here>' # a folder where training logs should be stored
```

# Reproducing Experiments

## Rosbag Download
Each localisation experiment requires a different bagfile with recorded sensor readings. Since each bagfile has 20-30 GB, we do not have a script that downloads all bagfiles in advance. Please download the bagfiles you need to a directory of choice and either

- provide the location of the downloaded bagfiles as an argument:
```bash
roslaunch background_foreground_segmentation <experiment>.launch rosbag_folder:=/your/rosbag/location
```
- OR link the folder once into the workspace
```bash
ln -s /your/rosbag/location $BFSEG_ROOT/rosbags
```

## Examples for localization experiments

Use an example bagfile (**Garage1**): [download bagfile](https://drive.google.com/file/d/1bVjDkZkycKaDUXlDpqKvr6hIjR4oC8ng/view?usp=sharing)

**Localisation without Segmentation**
```bash
roslaunch background_foreground_segmentation pickelhaube_full_garage1.launch
```
**Localisation with NYU-pretrained weights**
```bash
roslaunch background_foreground_segmentation pickelhaube_nyusegmentation_garage1.launch
```
**Localisation with Segmentation model trained in same environment (not on same dataset)**
```bash
roslaunch background_foreground_segmentation pickelhaube_segmentation_garage1.launch
```
# Software Overview

## Dataset Creator
## Overview
This package extracts the following information from a ROS Bag:

- For each camera:
    - Camera information (All information from the Sensor Message)
    - For each camera frame:
        - Timestamp
        - Camera pose in map as 4x4 Matrix [T_map_camera]
        - Camera Image
        - Point cloud in camera frame (as .pcd file, xyzi, where i is distance to mesh in mm)
        - Projected PointCloud in camera image (Only point cloud)
        - Projecctted PointCloud in camera image containing distance of point as grey value
        - Projecctted PointCloud in camera image containing distance to closest mesh as grey value


## Continual learning

### Overview

#### Dataset

- [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- Meshdist Dataset (our dataset generated by auto-labeling)
- Hive Dataset (with ground truth, for validation)

#### Methods

We implement the following methods for continual learning:

- Fine tuning (baseline)
- Feature distillation
- Output distillation
- [EWC](https://arxiv.org/pdf/1612.00796.pdf)

### Getting Started

Before you start, please modify the last line in [`Unet.py`](https://github.com/qubvel/segmentation_models/blob/94f624b7029deb463c859efbd92fa26f512b52b8/segmentation_models/models/unet.py#L252) from `"return model"` to `"return backbone, model"`.

**Experiment pipeline:**

**<span style="color:red">TODO</span>**
