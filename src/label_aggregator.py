#!/usr/bin/env python
"""
  ROS node that takes the sparse label of the projected pointcloud and aggregates them
"""
NAME = 'label_aggregator'

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

# Load config.
with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "../config/label_aggregator_cla.yaml"), 'r') as f:
  config = yaml.safe_load(f)

publishers = [
    rospy.Publisher(topic, Image, queue_size=10)
    for topic in config['outTopics']
]
labelOptions = config['labelOptions']


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

  aggregate_labels = aggregate_sparse_labels(
      np_labels,
      np_distance,
      np_original,
      outSize=(np_labels.shape[1] // labelOptions['downsamplingFactor'],
               np_labels.shape[0] // labelOptions['downsamplingFactor']),
      useSuperpixel=labelOptions['useSuperixel'],
      foregroundTrustRegion=labelOptions['foregroundTrustRegion'],
      fg_bg_threshold=labelOptions['fgBgThreshold'],
      superpixelCount=labelOptions['numberOfSuperPixel'])

  print(f"labeling took {(time.time() - t1):.4f}s")

  img_msg = Image()
  img_msg.header.seq = original.header.seq
  img_msg.header.frame_id = original.header.frame_id
  img_msg.header.stamp = original.header.stamp
  img_msg.height = aggregate_labels.shape[0]
  img_msg.width = aggregate_labels.shape[1]
  img_msg.step = original.width
  img_msg.data = aggregate_labels.ravel().tolist()
  img_msg.encoding = "mono8"

  return img_msg


def getCallbackForTopic(topicNumber):
  """ Callback wrapper.
    Returns a callback function that segements the image and publishs it

    Args:
      topicNumber: Number that speciefies to which topic the image should be published

  """

  def m_callback(*args):
    msg = callback(*args)
    publishers[topicNumber].publish(msg)

  return m_callback


def main():
  for idx, topics in enumerate(config['imageTopics']):
    originalTopic, labelsTopic, distanceTopic = (topics[0], topics[1],
                                                 topics[2])
    originalSub = message_filters.Subscriber(originalTopic, Image)
    labelsSub = message_filters.Subscriber(labelsTopic, Image)
    distanceSub = message_filters.Subscriber(distanceTopic, Image)

    ts = message_filters.ApproximateTimeSynchronizer(
        [originalSub, labelsSub, distanceSub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(getCallbackForTopic(idx))
    print("Subscribed to", originalTopic)

  rospy.init_node(NAME, anonymous=True)
  rospy.spin()


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt as e:
    pass
  print("exiting")
