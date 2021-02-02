# background_foreground_segmentation

structure allows for standalone python useage and useage within ROS.

To use the python package `cd src; pip install -e .`

## Dataset Creator
### Overview
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

### Getting Started
The following extracts all images from cam0 into the output folder <outputFolder>
1. Terminal A: `roscore`
2. Terminal B: `rosparam set use_sim_time true`
3. Terminal B: `roslaunch background_foreground_segmentation dataset_creator_standalone_rviz.launch output_folder:=<outputFolder>/ use_camera_stick:=cam0`
4. Terminal C: Play rosbag `rosbag play --clock <path/to/bagfile>`
5. RVIZ: Align Mesh with Pointcloud and right click marker -> load mesh, publish mesh

Sometimes due to bad timings running the standalone launch script can confuse the state estimator. In this case, start every node seperately:

1. Terminal A: `roscore`
2. Terminal B: `rosparam set use_sim_time true`
3. Terminal B: `roslaunch smb_state_estimator smb_state_estimator_standalone.launch`
4. Terminal C: `roslaunch cpt_selective_icp supermegabot_selective_icp_with_rviz.launch publish_distance:=true`
5. Terminal D: `roslaunch background_foreground_segmentation dataset_creator.launch outputFolder:=<outputFolder>/ use_camera_stick:=cam0`
5. Terminal E: `roslaunch segmentation_filtered_icp extrinsics.launch`
6. Terminal F: `rosbag play --clock <path/to/bagfile>`
5. RVIZ: Align Mesh with Pointcloud and right click marker -> load CAD, publish mesh
