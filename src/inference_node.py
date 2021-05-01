#!/usr/bin/env python
"""
  ROS node that segments the camera image using saved tensorflow models
"""
import sys
import rospy
import message_filters
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from zipfile import ZipFile
import os
import time
# Load model to predict foreground/background
import numpy as np
import tensorflow as tf
import gdown

from bfseg.settings import TMPDIR

tf.executing_eagerly()
# TODO: Not sure if required
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


def load_gdrive_file(file_id,
                     ending='',
                     output_folder=os.path.expanduser('~/.keras/datasets')):
  """Downloads files from google drive, caches files that are already downloaded."""
  filename = '{}.{}'.format(file_id, ending) if ending else file_id
  filename = os.path.join(output_folder, filename)
  if not os.path.exists(filename):
    gdown.download('https://drive.google.com/uc?id={}'.format(file_id),
                   filename,
                   quiet=False)
  return filename


def callback(pred_func, img_pubs, pointcloud, *image_msgs):
  """ Gets executed every time we get an image from the camera"""
  # Set headers for camera info
  startTime = time.time()
  # Get Images from message data
  imgs = []
  img_shapes = []
  img_headers = []
  for msg in image_msgs:
    img = np.frombuffer(msg.data, dtype=np.uint8)
    img = img.reshape(msg.height, msg.width, 3)
    # Convert BGR to RGB
    if 'bgr' in msg.encoding.lower():
      img = img[:, :, [2, 1, 0]]
    img_shapes.append((msg.height, msg.width))
    img_headers.append(msg.header)
    # resize to common input format
    img = tf.image.convert_image_dtype(tf.convert_to_tensor(img), tf.float32)
    img = tf.image.resize(
        img, (rospy.get_param('~input_height'), rospy.get_param('~input_width')))
    imgs.append(img)
  inputTime = time.time() 
  print("input time: {}".format(inputTime - startTime))
  # predict batch of images
  final_prediction = pred_func(tf.stack(imgs, axis=0))
  predictTime = time.time() 
  print("predict time: {}".format(predictTime - inputTime))
  for i, pred in enumerate(tf.unstack(final_prediction, axis=0)):
    # resize each prediction to the original image size
    prediction = tf.image.resize(pred[..., tf.newaxis], img_shapes[i],
                                 tf.image.ResizeMethod.BILINEAR)
    # convert to numpy
    prediction = prediction.numpy().astype('uint8')
    # Create and publish image message
    img_msg = Image()
    img_msg.header = img_headers[i]
    img_msg.height = img_shapes[i][0]
    img_msg.width = img_shapes[i][1]
    img_msg.step = img_msg.width
    img_msg.data = prediction.flatten().tolist()
    img_msg.encoding = "mono8"
    img_pubs[i].publish(img_msg)
  outputTime = time.time()
  print("output time: {}".format(outputTime - predictTime))
  timeDiff = time.time() - startTime
  print("published segmented images in {:.4f}s, {:.4f} FPs".format(
      timeDiff, 1 / timeDiff))


def main_loop():
  rospy.init_node('inference_node')
  image_subscribers = [
      message_filters.Subscriber(topic, Image)
      for topic in rospy.get_param('~image_topics')
  ]
  lidar_subscriber = message_filters.Subscriber(
      rospy.get_param('~pointcloud_topic'), PointCloud2)
  # publishers for segmentation maps
  img_pubs = [
      rospy.Publisher(topic, Image, queue_size=10)
      for topic in rospy.get_param('~segmentation_output_topics')
  ]
  # load the  model
  ZipFile(load_gdrive_file(rospy.get_param('~model_gdrive_id'),
                           ending='zip')).extractall(
                               os.path.join(TMPDIR, 'segmentation_model'))
  model = tf.saved_model.load(os.path.join(TMPDIR, 'segmentation_model'))

  @tf.function
  def pred_func(batch):
    # predict batch of images
    return tf.squeeze(tf.nn.softmax(model(batch), axis=-1)[..., 1] * 255)

  # only get those images that will be synchronized to the lidar
  synchronizer = message_filters.ApproximateTimeSynchronizer(
      [lidar_subscriber] + image_subscribers, 10, 0.1)
  synchronizer.registerCallback(lambda *x: callback(pred_func, img_pubs, *x))
  rospy.spin()


if __name__ == '__main__':
  try:
    main_loop()
  except KeyboardInterrupt as e:
    pass
  print("exiting")
