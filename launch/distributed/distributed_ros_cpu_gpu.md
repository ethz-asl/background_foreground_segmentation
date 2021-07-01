## Running ROS in distributed Fashion

Open the following shells:



CPU 1: 
```bash
cd bfs_catkin_ws
source devel/setup.bash
roscore
# read ROS_MASTER_URI (here: http://192.168.0.102:11311 )
```

CPU 2: 
```bash
conda activate bfs1 # or whatever your environment is called
cd bfs_catkin_ws
source devel/setup.bash
export ROS_IP=192.168.0.102 # replace with CPU IP
export ROS_MASTER_URI=http://192.168.0.102:11311 # replace with own one
roslaunch src/background_foreground_segmentation/launch/distributed/pickelhaube_online_learning_rumlang1_cpu.launch
```

GPU 1: 
```bash
ssh xavier@<IPAdress>
source envs/bfs1/bin/activate
cd bfs_catkin_ws 
source devel/setup.bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 # for online_learning imports
export ROS_IP=192.168.0.106 # replace with Xavier IP
export ROS_MASTER_URI=http://192.168.0.102:11311 # replace with own one 
roslaunch src/gpu_learning/launch/distributed/pickelhaube_online_learning_rumlang1_gpu.launch
```
