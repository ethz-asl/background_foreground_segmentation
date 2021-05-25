import numpy as np
from scipy.spatial.transform import Rotation as R

CAMCAM = True

T_rgb_base = np.array([[-0.0020120, -0.9999900,  0.0039939, -0.032],
   [0.0059959, -0.0040059, -0.9999740, 0.0],
   [0.9999800, -0.0019880,  0.0060039, 0.004],
   [0, 0, 0, 1]]) # from tf tf_echo pickelhaubergb_camera_link pickelhaubecamera_base

if True:
    T_rgb_imu = np.array([[0.9999629813933317, 0.0002329962865674027, 0.008601253146517615, 0.059294779378346034],
    [-0.008601631877306825, 0.0016764133435310247, 0.9999616000463967, -0.007137508332827874],
    [0.0002185680839748988, -0.9999985676744907, 0.0016783554333306694, -0.09219845857111238],
    [0.0, 0.0, 0.0, 1.0]]) # from camIMU kalibr

    T_imu_base = np.matmul(np.linalg.inv(T_rgb_imu), T_rgb_base)
    print("T_imu_base: ")
    print(T_imu_base)
    r = R.from_matrix(T_imu_base[0:3, 0:3]).as_quat()
    t = T_imu_base[0:3, 3]
    print("Trans/Quat: IMUFRAME pickelhaubecamera_base")
    print(t)
    print(r)
if True:
    T_rgb_cam2 = np.array([[0.9999553225604283, -0.009414474055130075, -0.0008488588438938574, 0.10583807977314026],
    [0.00941326842796873, 0.9999546954914059, -0.001413273573806863, -0.05672334028949535],
    [0.0008621256141544741, 0.0014052198962071418, 0.9999986410473118, -0.035754646718517115],
    [0.0, 0.0, 0.0, 1.0]]) # from camcam kalibr

    T_cam2_base = np.matmul(np.linalg.inv(T_rgb_cam2), T_rgb_base)
    print("T_cam2_base: ")
    print(T_cam2_base)
    r = R.from_matrix(T_cam2_base[0:3, 0:3]).as_quat()
    t = T_cam2_base[0:3, 3]
    print("Trans/Quat: cam2 pickelhaubecamera_base")
    print(t)
    print(r)