# Self Supervised Segmentation
1. ### Dataset Creator (dataset_creator):
    Ros node that extracts the following information from a .bag file:
    - Name of the ROS bag and corresponding mesh file
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

2. ### Image preprocessor (image_preprocessor):
    Jupyter notebook file that converts sparse annotations given by projected point cloud into bigger annotations using superpixels or region growing

3. ### Meshdistance Labler (Meshdist labler) - DEPRECATED - TO BE REMOVED:
    Ros node that calculates distance to mesh for each point of a pointcloud and projecting them into camera frame.
    Functionality is now inside Dataset Creator node

4. ### Neural Network Training (training):
    Code to train neural network using sparse annotations given by the image preprocessor