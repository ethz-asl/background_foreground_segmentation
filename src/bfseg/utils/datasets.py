""" Utils to pre-process and load the datasets.
"""
import tensorflow as tf
import tensorflow_datasets as tfds


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


def load_data(dataset_name, mode, batch_size, scene_type):
  r"""Creates a data loader given the dataset parameters as input.
  TODO(fmilano): Check this whole function.

  Args:
    dataset_name (str): Name of the dataset. Valid entries are:
      "NyuDepthV2Labeled" (NYU dataset), "BfsegCLAMeshdistLabels".
    mode (str): Identifies the type of dataset. Valid entries are: "train",
      "val", "test".
    batch_size (int): Batch size.
    scene_type (str): Scene type. Valid entries are: None, "kitchen", "bedroom".

  Returns:
    ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Data loader for
      the dataset with input parameters.
  """
  if (dataset_name == 'NyuDepthV2Labeled'):
    if (scene_type == None):
      name = 'full'
    elif (scene_type == "kitchen"):
      name = 'train'
    elif (scene_type == "bedroom"):
      name = 'test'
    else:
      raise Exception("Invalid scene type: %s!" % scene_type)
  elif (dataset_name == 'BfsegCLAMeshdistLabels'):
    name = 'fused'
  else:
    raise Exception("Dataset %s not found!" % dataset_name)
  if (mode == 'train'):
    split = name + '[:80%]'
    shuffle = True
  else:
    split = name + '[80%:]'
    shuffle = False
  ds, info = tfds.load(
      dataset_name,
      split=split,
      shuffle_files=shuffle,
      as_supervised=True,
      with_info=True,
  )
  if (dataset_name == 'NyuDepthV2Labeled'):
    ds = ds.map(preprocess_nyu,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif (dataset_name == 'BfsegCLAMeshdistLabels'):
    ds = ds.map(preprocess_cla,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.cache()
  if (mode == 'train'):
    ds = ds.shuffle(int(info.splits[name].num_examples * 0.8))
  ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
  return ds
