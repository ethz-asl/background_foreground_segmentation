#!/usr/bin/env python3
"""
  ROS node that segments the camera image using online learned model
  trained on pseudolabels.
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import rospy
import message_filters
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import gdown
import time
import numpy as np
import random

from bfseg.utils.models import create_model
from bfseg.utils.losses import BalancedIgnorantCrossEntropyLoss
from bfseg.utils.image_enhancement import aggregate_sparse_labels
from bfseg.utils.images import augmentation

tf.executing_eagerly()

nyu_data = iter(
    tfds.load('nyu_subsampled', split='full',
              as_supervised=True).cache().repeat().map(augmentation))

LABELED_BUFFER = []

INPUT_HEIGHT = 480
INPUT_WIDTH = 640


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


def callback(pred_func, img_pubs, checkpointer, pointcloud, *image_msgs):
  """ Gets executed every time we get an image from the camera"""
  # Set headers for camera info
  startTime = time.time()
  # Get Images from message data
  input_batch = []
  img_shapes = []
  img_headers = []
  for i, msg in enumerate(image_msgs):
    img = np.frombuffer(msg.data, dtype=np.uint8)
    img = img.reshape(msg.height, msg.width, 3)
    # Convert BGR to RGB
    if 'bgr' in msg.encoding.lower():
      img = img[:, :, [2, 1, 0]]
    img_shapes.append((msg.height, msg.width))
    img_headers.append(msg.header)
    # resize to common input format
    img = tf.image.convert_image_dtype(tf.convert_to_tensor(img), tf.float32)
    img = tf.image.resize(img, (INPUT_HEIGHT, INPUT_WIDTH))
    input_batch.append(img)

  # fill batch with training data
  # 3 for inference, 9 replaying from the past, 1 (10%) replaying from NYU
  to_be_used = random.sample(LABELED_BUFFER, min(9, len(LABELED_BUFFER)))
  input_batch.extend([elem['image'] for elem in to_be_used])
  label_batch = [elem['label'] for elem in to_be_used]
  nyu_sample = next(nyu_data)
  input_batch.append(
      tf.image.convert_image_dtype(nyu_sample[0], tf.float32))
  label_batch.append(nyu_sample[1])
  # predict batch of images
  final_prediction = pred_func(tf.stack(input_batch, axis=0),
                               tf.stack(label_batch, axis=0))
  # save checkpoint
  checkpointer.save()
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

  timeDiff = time.time() - startTime
  print("published segmented images in {:.4f}s, {:.4f} FPs".format(
      timeDiff, 1 / timeDiff))


def _label_callback(original, labels, distance):
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
      outSize=(np_labels.shape[1], np_labels.shape[0]),
      useSuperpixel=options['useSuperpixel'],
      foregroundTrustRegion=options['foregroundTrustRegion'],
      fg_bg_threshold=options['fgBgThreshold'],
      superpixelCount=options['numberOfSuperPixel'])
  # swap classes 1 and 2
  labelmap = np.array([0, 2, 1])
  aggregate_labels = labelmap[aggregate_labels].astype('uint8')

  print("labeling took {:.4f}s".format(time.time() - t1))

  # throw data away if buffer is too full
  if len(LABELED_BUFFER) > 500:
    new_buffer = random.sample(LABELED_BUFFER, 250)
    LABELED_BUFFER.clear()
    LABELED_BUFFER.extend(new_buffer)

  LABELED_BUFFER.append({
      'image':
          tf.image.resize(
              tf.image.convert_image_dtype(tf.convert_to_tensor(np_original),
                                           tf.float32),
              (INPUT_HEIGHT, INPUT_WIDTH)),
      'label':
          tf.image.resize(tf.convert_to_tensor(aggregate_labels)[...,
                                                                 tf.newaxis],
                          (INPUT_HEIGHT, INPUT_WIDTH),
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
  })


def label_callback(topicNumber, counters):
  """ Callback wrapper.
    Returns a callback function that segements the image and publishs it

    Args:
      topicNumber: Number that speciefies to which topic the image should be published
  """

  def m_callback(*args):
    # has to access nonlocal object, integers don't work in py2
    counters[topicNumber] += 1
    if not counters[topicNumber] % rospy.get_param('~label_frequency', 1) == 0:
      return
    _label_callback(*args)

  return m_callback


def main_loop():
  print('START', flush=True)
  rospy.init_node('inference_node')
  print('ROSPY', flush=True)
  image_subscribers = [
      message_filters.Subscriber(topics[0], Image)
      for topics in rospy.get_param('~image_topics')
  ]
  lidar_subscriber = message_filters.Subscriber(
      rospy.get_param('~pointcloud_topic'), PointCloud2)
  # publishers for segmentation maps
  img_pubs = [
      rospy.Publisher(topic, Image, queue_size=10)
      for topic in rospy.get_param('~segmentation_output_topics')
  ]
  # load the  model
  _, full_model = create_model(model_name="fast_scnn",
                               freeze_encoder=False,
                               freeze_whole_model=False,
                               normalization_type="group",
                               image_h=INPUT_HEIGHT,
                               image_w=INPUT_WIDTH)
  model = tf.keras.Model(inputs=full_model.input, outputs=full_model.output)
  weights_path = load_gdrive_file(rospy.get_param('~model_gdrive_id'),
                                  ending='h5')
  model.load_weights(weights_path)
  lossfunc = BalancedIgnorantCrossEntropyLoss(class_to_ignore=2,
                                              num_classes=3,
                                              from_logits=True)
  optimizer = tf.keras.optimizers.Adam(rospy.get_param('~learning_rate'))

  print('MODEL LOADED', flush=True)

  folder_counter = 1
  checkpoints_folder = '/home/blumh/asl/rss_2021_data/online_learning_{}'
  while os.path.exists(checkpoints_folder.format(folder_counter)):
    folder_counter += 1
  checkpoints_folder = checkpoints_folder.format(folder_counter)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint_step = tf.Variable(0)
  manager = tf.train.CheckpointManager(
    checkpoint,
    directory=checkpoints_folder,
    max_to_keep=None,
    step_counter=checkpoint_step,
    checkpoint_interval=10)
  # immidiately create a first checkpoint
  manager.save()

  @tf.function
  def pred_func(inputs, labels):
    """ predict batch of images and update model with loss. The first len(img_pubs)
    images are returned as prediction, the rest of the batch can be just any images to
    learn on.
    """
    nonlocal checkpoint_step
    checkpoint_step.assign_add(1)
    with tf.GradientTape() as tape:
      logits = model(inputs)
      pred = tf.squeeze(
          tf.nn.softmax(logits[:len(img_pubs)], axis=-1)[..., 1] * 255)
      loss = lossfunc(labels, logits[len(img_pubs):])
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return pred

  # only get those images that will be synchronized to the lidar
  synchronizer = message_filters.ApproximateTimeSynchronizer(
      [lidar_subscriber] + image_subscribers, 10, 0.1)
  synchronizer.registerCallback(lambda *x: callback(pred_func, img_pubs, manager,  *x))

  # other callbacks to collect pseudolabels
  counters = [0 for topics in rospy.get_param('~image_topics')]
  for idx, topics in enumerate(rospy.get_param('~image_topics')):
    originalTopic, labelsTopic, distanceTopic = (topics[0], topics[1],
                                                 topics[2])
    originalSub = message_filters.Subscriber(originalTopic, Image)
    labelsSub = message_filters.Subscriber(labelsTopic, Image)
    distanceSub = message_filters.Subscriber(distanceTopic, Image)

    ts = message_filters.ApproximateTimeSynchronizer(
        [originalSub, labelsSub, distanceSub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(label_callback(idx, counters))
  rospy.spin()


if __name__ == '__main__':
  try:
    main_loop()
  except KeyboardInterrupt as e:
    pass
  print("exiting")
