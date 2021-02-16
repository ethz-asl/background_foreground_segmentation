#!/usr/bin/env python
"""
  ROS node that takes the sparse label of the projected pointcloud and aggregates them
"""
import sys
import message_filters
import rospy
from std_msgs.msg import *
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image
import yaml
import time
from bfseg.utils.image_enhancement import aggregate_sparse_labels
import numpy as np
import os
import cv2


def callback(original, labels, distance):
  """ Returns the pseudo labeled image """

  np_original = np.frombuffer(original.data,
                              dtype=np.uint8).reshape(original.height,
                                                      original.width, -1)
  np_labels = np.frombuffer(labels.data,
                            dtype=np.uint8).reshape(labels.height, labels.width)
  np_distance = np.frombuffer(distance.data,
                              dtype=np.uint8).reshape(distance.height,
                                                      distance.width)
  t1 = time.time()

  options = rospy.get_param('~label_options')
  aggregate_labels = aggregate_sparse_labels(
      np_labels,
      np_distance,
      np_original,
      outSize=(np_labels.shape[1] // options['downsamplingFactor'],
               np_labels.shape[0] // options['downsamplingFactor']),
      useSuperpixel=options['useSuperpixel'],
      foregroundTrustRegion=options['foregroundTrustRegion'],
      fg_bg_threshold=options['fgBgThreshold'],
      superpixelCount=options['numberOfSuperPixel'])
  # swap classes 1 and 2
  labelmap = np.array([0, 2, 1])
  aggregate_labels = labelmap[aggregate_labels]

  print("labeling took {:.4f}s".format(time.time() - t1))

  # downsampling of original image
  resized_original = cv2.resize(
      np_original, (np_original.shape[1] // options['downsamplingFactor'],
                    np_original.shape[0] // options['downsamplingFactor']))

  img_msg = Image()
  img_msg.header.seq = original.header.seq
  img_msg.header.frame_id = original.header.frame_id
  img_msg.header.stamp = original.header.stamp
  img_msg.height = aggregate_labels.shape[0]
  img_msg.width = aggregate_labels.shape[1]
  img_msg.step = original.width
  img_msg.data = aggregate_labels.ravel().tolist()
  img_msg.encoding = "mono8"

  return img_msg, aggregate_labels, resized_original, original.encoding


def getCallbackForTopic(topicNumber, publisher, counters):
  """ Callback wrapper.
    Returns a callback function that segements the image and publishs it

    Args:
      topicNumber: Number that speciefies to which topic the image should be published
      publisher: the publisher used
  """

  def m_callback(*args):
    # has to access nonlocal object, integers don't work in py2
    counters[topicNumber] += 1
    if not counters[topicNumber] % rospy.get_param('~label_frequency', 1) == 0:
      return
    msg, labels, img, img_enc = callback(*args)
    if rospy.get_param('~publish_labels', False):
      publisher.publish(msg)
    if rospy.get_param('~store_labels', False):
      cv2.imwrite(
          os.path.join(
              rospy.get_param('~label_path'),
              '{}_cam{}_labels.png'.format(msg.header.stamp, topicNumber)),
          labels.astype('uint8'))
      # convert to BGR for opencv
      if 'rgb' in img_enc:
        img = img[..., ::-1]
      cv2.imwrite(
          os.path.join(rospy.get_param('~label_path'),
                       '{}_cam{}_rgb.png'.format(msg.header.stamp,
                                                 topicNumber)), img)

  return m_callback


def main():
  rospy.init_node('label_aggregator')
  # prepare directory
  if rospy.get_param('~store_labels', False) and not os.path.exists(
      rospy.get_param('~label_path')):
    os.mkdir(rospy.get_param('~label_path'))

  # prepare publishers
  publishers = [
      rospy.Publisher(topic, Image, queue_size=10)
      for topic in rospy.get_param('~out_topics')
  ]
  counters = [0 for topic in rospy.get_param('~out_topics')]

  for idx, topics in enumerate(rospy.get_param('~image_topics')):
    originalTopic, labelsTopic, distanceTopic = (topics[0], topics[1],
                                                 topics[2])
    originalSub = message_filters.Subscriber(originalTopic, Image)
    labelsSub = message_filters.Subscriber(labelsTopic, Image)
    distanceSub = message_filters.Subscriber(distanceTopic, Image)

    ts = message_filters.ApproximateTimeSynchronizer(
        [originalSub, labelsSub, distanceSub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(getCallbackForTopic(idx, publishers[idx], counters))
    print("Subscribed to", originalTopic)

  rospy.spin()


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt as e:
    pass
  print("exiting")
