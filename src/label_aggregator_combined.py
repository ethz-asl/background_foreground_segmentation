#!/usr/bin/env python
"""
  ROS node that takes the sparse label of the projected pointcloud and aggregates them
"""
import sys
import rospy
import os
from sensor_msgs.msg import CameraInfo, Image
import message_filters
import numpy as np
import time
import cv2

from bfseg.utils.image_enhancement import aggregate_sparse_labels


def callback(original, labels, distance, labels_dense):
  """ Returns the pseudo labeled image """

  np_original = np.frombuffer(original.data,
                              dtype=np.uint8).reshape(original.height,
                                                      original.width, -1)
  np_labels = np.frombuffer(labels.data,
                            dtype=np.uint8).reshape(labels.height, labels.width)
  np_distance = np.frombuffer(distance.data,
                              dtype=np.uint8).reshape(distance.height,
                                                      distance.width)
  np_labels_dense = np.frombuffer(labels_dense.data,
                            dtype=np.uint8).reshape(labels.height, labels.width)
  
  t1 = time.time()

  options = rospy.get_param('~label_options')
  if options['agreement'] or options['useSuperpixel']:
    superpixels = aggregate_sparse_labels(
        np_labels,
        np_distance,
        np_original,
        outSize=(np_labels.shape[1] // options['downsamplingFactor'],
                 np_labels.shape[0] // options['downsamplingFactor']),
        stdDevThreshold=options.get('stdDevThreshold', 0.5),
        useSuperpixel=True,
        foregroundTrustRegion=options['foregroundTrustRegion'],
        fg_bg_threshold=options['fgBgThreshold'],
        superpixelCount=options['numberOfSuperPixel'])
  if options['agreement'] or not options['useSuperpixel']:
    region_growing = aggregate_sparse_labels(
        np_labels,
        np_distance,
        np_original,
        outSize=(np_labels.shape[1] // options['downsamplingFactor'],
                np_labels.shape[0] // options['downsamplingFactor']),
        stdDevThreshold=options.get('stdDevThreshold', 0.5),
        useSuperpixel=False,
        foregroundTrustRegion=options['foregroundTrustRegion'],
        fg_bg_threshold=options['fgBgThreshold'],
        superpixelCount=options['numberOfSuperPixel'])

  if options['agreement']:
    aggregate_labels = np.where(superpixels == region_growing, superpixels,
                                np.ones_like(superpixels))
  elif options['useSuperpixel']:
    aggregate_labels = superpixels
  else:
    aggregate_labels = region_growing
  # swap classes 1 and 2
  labelmap = np.array([0, 2, 1])
  aggregate_labels = labelmap[aggregate_labels]

  print("labeling took {:.4f}s".format(time.time() - t1))

  # downsampling of original image
  resized_original = cv2.resize(
      np_original, (np_original.shape[1] // options['downsamplingFactor'],
                    np_original.shape[0] // options['downsamplingFactor']))

  # combine dense label with sparse labels: take dense labels whereever they are defined
  aggregate_labels = np.where(np_labels_dense == 2, aggregate_labels, np_labels_dense)

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
      print("Writing to {}...".format(rospy.get_param('~label_path')))
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
  rospy.init_node('label_aggregator_combined')
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
    originalTopic, labelsTopic, distanceTopic, denseLabelsTopic = (topics[0], topics[1],
                                                 topics[2], topics[3])
    originalSub = message_filters.Subscriber(originalTopic, Image)
    labelsSub = message_filters.Subscriber(labelsTopic, Image)
    distanceSub = message_filters.Subscriber(distanceTopic, Image)
    denseLabelsSub = message_filters.Subscriber(denseLabelsTopic, Image)

    ts = message_filters.ApproximateTimeSynchronizer(
        [originalSub, labelsSub, distanceSub, denseLabelsSub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(getCallbackForTopic(idx, publishers[idx], counters))
    print("Subscribed to", originalTopic)
  rospy.spin()


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt as e:
    pass
  print("exiting")
