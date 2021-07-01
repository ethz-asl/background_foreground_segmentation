# Setup Background Foreground Segmentation on Ubuntu 20.04

This is the setup that worked for me, I hope it can help to troubleshoot any issues that come up for someone else.

Create virtual environment
```bash
conda create -n bfs1 python=3.6
conda activate bfs1
cd ~/git/background_foreground_segmentation
pip install -r requirements.txt
pip install catkin_pkg (not in requirements)
```
If not done: install ROS as in http://wiki.ros.org/noetic/Installation/Ubuntu

If not done: install catkin_tools according to https://catkin-tools.readthedocs.io/en/latest/installing.html#installing-on-ubuntu-with-apt-get 


Setup Catkin Workspace
```bash
mkdir -p ~/bfs_catkin_ws/src
cd ~/bfs_catkin_ws
catkin init
ln -s ~/git/background_foreground_segmentation/ ~/bfs_catkin_ws/src/ #symlink to git repo
# # use wstool to get dependencies from git (not sure if necessary)
sudo pip install -U wstool
wstool init src ~/git/background_forground_segmentation/dependencies.rosinstall
cd src
```

For later changes in dependencies: 
```bash
wstool merge ~/git/background_forground_segmentation/dependencies.rosinstall
wstool update 
```

Fixes for errors occuring during Installs:
```bash
# Missing packages 
pip install empy # doesn't suffice to do it via apt-get python3-empy 
sudo apt-get install libgmp3-dev
sudo apt-get install libmpfr3-dev
sudo apt install ros-noetic-geometric-shapes
pip install rospkg
# Missing opencv (C++), therefore installing it from source
cd ~/git
git clone https://github.com/opencv/opencv.git
mkdir -p build 
cd build
cmake ../opencv
make -j4
# Need opencv3, therefore installing it from source (opencv4 in /usr/include/, new opencv and opencv2 in /usr/local/include)
cd ~/git 
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.4.13 
mkdir build
cd build 
cmake .. -D CMAKE_INSTALL_PREFIX=/usr/local
make -j4
sudo make install
sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
```

Build packages with catkin build
```bash
catkin config --cmake-args -DCMAKE_CXX_STANDARD=14 -DCMAKE_BUILD_TYPE=Release # necessary for image_undistort, fix proposed by https://github.com/ros/catkin/issues/936
catkin build
source devel/setup.bash
```