#!/usr/bin/env python
"""
  ROS node that publishes camera intrinsics and if enabled also segments the camera image
"""
NAME = 'image_and_camera_publisher'

import sys
import rospy
from std_msgs.msg import *
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image
import yaml
import time

with open(
    "/home/rene/catkin_ws/src/background_foreground_segmentation/config/ros_image_publisher_default.yaml",
    'r') as f:
  config = yaml.safe_load(f)

# Model used to predict background foreground segmentation
model = None
# Publisher to publish segmented images
img_pub = None

pub = rospy.Publisher('/rgb/camera_info', CameraInfo, queue_size=10)

# Camera info from kinect camera
c_info = CameraInfo(height=config['cam']['height'],
                    width=config['cam']['width'],
                    distortion_model=config['cam']['distortion_model'],
                    D=config['cam']['D'],
                    K=config['cam']['K'],
                    R=config['cam']['R'],
                    P=config['cam']['P'])

if config['processImages']:
  # Load model to predict foreground/background
  import numpy as np
  import tensorflow as tf

  tf.executing_eagerly()

  img_pub = rospy.Publisher(config['segmentationOutTopic'],
                            Image,
                            queue_size=10)
  model = tf.keras.models.load_model(config['pathToModel'],
                                     custom_objects={'tf': tf})


def callback(msg, *args):
  """ Gets executed every time we get an image from the camera"""
  # Set headers for camera info
  c_info.header = msg.header
  c_info.header.frame_id = config['frameId']
  c_info.header.seq = msg.header.seq
  pub.publish(c_info)
  print("published camera parameters")

  if model is not None:
    startTime = time.time()
    # Get Image from message data
    img = np.frombuffer(msg.data, dtype=np.uint8)
    # Convert BGR to RGB
    img = img.reshape(msg.height, msg.width, 4)[:, :, [2, 1, 0]]
    # Rotate image
    img = tf.convert_to_tensor(np.rot90(np.rot90(img)))
    img_shape = img.shape

    # scale to input size
    img = tf.image.resize(tf.cast(img, dtype=float) / 255, [480, 640])
    # predict image
    pred = tf.squeeze(
        tf.argmax(model.predict(tf.expand_dims(img, axis=0))[1], axis=-1))
    # scale to output size, change dtype
    final_prediction = tf.image.resize(
        tf.expand_dims(pred * 255, axis=-1), [img_shape[0], img_shape[1]],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy().astype(np.uint8)
    # rotate back
    final_prediction = np.rot90(np.rot90(final_prediction))

    # Create and publish image message
    img_msg = Image()
    img_msg.header.seq = c_info.header.seq
    img_msg.header.frame_id = c_info.header.frame_id
    img_msg.header.stamp = c_info.header.stamp
    img_msg.height = img_shape[0]
    img_msg.width = img_shape[1]
    img_msg.step = msg.width
    img_msg.data = final_prediction.flatten().tolist()
    img_msg.encoding = "mono8"
    img_pub.publish(img_msg)

    timeDiff = time.time() - startTime
    print("published segmented image in {:.4f}s, {:.4f} FPs".format(
        timeDiff, 1 / timeDiff))


def main_loop():
  rospy.Subscriber(config["imageTopic"], Image, callback, 1)
  rospy.init_node(NAME, anonymous=True)
  rospy.spin()


if __name__ == '__main__':
  try:
    main_loop()
  except KeyboardInterrupt as e:
    pass
  print("exiting")
