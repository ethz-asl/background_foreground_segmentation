""" Utils to pre-process the datasets.
"""
import tensorflow as tf


@tf.function
def preprocess_nyu(image, label):
  r"""Preprocesses NYU dataset:
  - Normalize images: `uint8` -> `float32`.
  - Assigns label: 1 if belong to background, 0 if foreground.
  - Creates all-True mask, since NYU labels are all known.
  """
  mask = tf.not_equal(label, -1)  # All true.
  label = tf.expand_dims(label, axis=2)
  image = tf.cast(image, tf.float32) / 255.
  return image, label, mask


@tf.function
def preprocess_cla(image, label):
  r"""Preprocesses our auto-labeled CLA dataset. The dataset consists of three
  labels (0,1,2) where all classes that belong to the background (e.g., floor,
  wall, roof) are assigned the '2' label. Foreground is assigned the '0' label
  and unknown the '1' label. To let CLA label format be consistent with NYU, for
  each input pixel the label outputted by this function is assigned as follows:
  1 if the pixel belongs to the background, 0 if it belongs to foreground /
  unknown (does not matter since we are using masked loss). An element in the
  output mask is True if the corresponding pixel is of a known class (label '0'
  or '2'). The mask is used to computed the masked cross-entropy loss.
  """
  mask = tf.squeeze(tf.not_equal(label, 1))
  label = tf.cast(label == 2, tf.uint8)
  image = tf.cast(image, tf.float32)
  return image, label, mask
