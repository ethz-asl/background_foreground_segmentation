# Xavier Setup

Set up Xavier via NVIDIA SDK Manager (need a computer with Ubuntu 18.04 for that). 
Watch GPU performance with `tegrastats` command.
Controll fan with inputs from 0 (min) to 255 (max) via `echo 255 | sudo tee /sys/devices/pwm-fan/target_pwm`

## Install Tensorflow and other stuff

```bash
# get Tensorflow (https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)
# possible issues when Tensorlow installation fails: https://github.com/jneilliii/OctoPrint-BedLevelVisualizer/issues/141 // https://forums.developer.nvidia.com/t/installing-python-packages-with-pip/72853 (installing TF before CV2 gives an error!)
sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install python3-pip
sudo pip3
sudo -H pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11 install -U pip testresources setuptools==49.6.0
pip3 install cython
pip3 install gdown
sudo /usr/bin/jetson_clocks 
pip3 install sacred yapf pylint pytest tensorflow-datasets
sudo apt-get install python3-opencv
pip3 install --upgrade pip
pip3 install scikit-build
pip3 install opencv-python==4.3.0.38 --no-cache-dir
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow==2.3.1+nv20.12 --no-cache-dir
# install tensorflow addons according to https://qengineering.eu/install-tensorflow-addons-on-jetson-nano.html 
wget https://github.com/Qengineering/TensorFlow-Addons-Jetson-Nano/raw/main/tensorflow_addons-0.13.0.dev0-cp36-cp36m-linux_aarch64.whl
pip3 install tensorflow_addons-0.13.0.dev0-cp36-cp36m-linux_aarch64.whl
pip3 install segmentation-models and rest from requirements.txt
```
To run trainings, have a look at experiments_cfg/dense_label_experiments.md. 

## Install ROS 

Install ROS Melodic on GPU (https://developer.ridgerun.com/wiki/index.php?title=Robot_Operating_System_ROS_on_Jetson_Platforms)
```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-melodic-desktop
sudo apt-get install python-rosdep -y
sudo apt-get install python-rosinstall python-rosinstall-generator python-wstool build-essential -y
sudo c_rehash /etc/ssl/certs
sudo rosdep init
rosdep update
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
# Install catkin tools (same as for normal setup)
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'
wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install python-catkin-tools
# Setup Catkin Workspace
mkdir -p ~/bfs_catkin_ws/src
cd ~/bfs_catkin_ws
catkin init
# Create symlink to git repo 
ln -s ~/git/background_foreground_segmentation/ ~/bfs_catkin_ws/src/
# look at minimal rosnode that I stored there with symlinks to src and launch folders!
pip install rospkg
catkin build
# very nice fix for keras-tuner from https://github.com/keras-team/keras-tuner/issues/317
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 
```

