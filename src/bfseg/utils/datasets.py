""" Utils to pre-process and load the datasets.
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import warnings


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


def load_data(dataset_name, scene_type, fraction, batch_size, shuffle_data):
  r"""Creates a data loader given the dataset parameters as input.

  Args:
    dataset_name (str): Name of the dataset. Valid entries are:
      "NyuDepthV2Labeled" (NYU dataset), "BfsegCLAMeshdistLabels".
    scene_type (str): Type of scene in the dataset to use type. Valid entries
      are:
      - If `dataset_name` is "NyuDepthV2Labeled":
        - None: All the scenes in the dataset are selected.
        - "kitchen": Kitchen scene only.
        - "bedroom": Bedroom scene only.
      - If `dataset_name` is "BfsegCLAMeshdistLabels":
        - None: All the samples in the dataset are selected (no scene
            subdivision is available).
    fraction (str): Fraction of the selected scene to load. Must be a valid
      slice (cf. https://www.tensorflow.org/datasets/splits), e.g., "[:80%]"
      (first 80% of the samples in the scene), "[80%:]" (last 20% of the samples
      in the scene), or None (all samples in the scene are selected).
    batch_size (int): Batch size.
    shuffle_data (bool): Whether or not to shuffle data.

  Returns:
    ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Data loader for
      the dataset with input parameters.
  """
  # Select the scene.
  if (dataset_name == 'NyuDepthV2Labeled'):
    if (scene_type == None):
      name = 'full'
    elif (scene_type == "kitchen"):
      # The "kitchen" scene is referred to as "train" split in NYU.
      name = 'train'
    elif (scene_type == "bedroom"):
      # The "bedroom" scene is referred to as "test" split in NYU.
      name = 'test'
    else:
      raise Exception("Invalid scene type: %s!" % scene_type)
  elif (dataset_name == 'BfsegCLAMeshdistLabels'):
    if (scene_type is None):
      # The CLA dataset contains a single split called "fused".
      name = 'fused'
    else:
      raise Exception("Invalid scene type: %s!" % scene_type)
  else:
    raise Exception("Dataset %s not found!" % dataset_name)

  # Select the fraction of samples from the scene.
  if (fraction is not None):
    split = f"{name}{fraction}"
  else:
    split = name

  # Actually load the dataset.
  ds = tfds.load(dataset_name,
                 split=split,
                 shuffle_files=shuffle_data,
                 as_supervised=True)
  # Apply pre-processing.
  if (dataset_name == 'NyuDepthV2Labeled'):
    ds = ds.map(preprocess_nyu,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif (dataset_name == 'BfsegCLAMeshdistLabels'):
    ds = ds.map(preprocess_cla,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.cache()
  # Further shuffle.
  if (shuffle_data):
    ds = ds.shuffle(len(ds))

  ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

  return ds


def load_datasets(train_dataset,
                  train_scene,
                  test_dataset,
                  test_scene,
                  batch_size,
                  validation_percentage,
                  fisher_params_dataset=None,
                  fisher_params_scene=None,
                  fisher_params_sample_percentage=None):
  r"""Creates 3 data loaders, for training, validation and testing.
    """
  assert (isinstance(validation_percentage, int) and
          0 <= validation_percentage <= 100)
  assert ((fisher_params_dataset is None) == (fisher_params_scene is None) ==
          (fisher_params_sample_percentage is None))
  training_percentage = 100 - validation_percentage
  train_ds = load_data(dataset_name=train_dataset,
                       scene_type=train_scene,
                       fraction=f"[:{training_percentage}%]",
                       batch_size=batch_size,
                       shuffle_data=True)
  val_ds = load_data(dataset_name=train_dataset,
                     scene_type=train_scene,
                     fraction=f"[{training_percentage}%:]",
                     batch_size=batch_size,
                     shuffle_data=False)
  test_ds = load_data(dataset_name=test_dataset,
                      scene_type=test_scene,
                      fraction=None,
                      batch_size=batch_size,
                      shuffle_data=False)
  if (fisher_params_dataset is None):
    return train_ds, val_ds, test_ds
  else:
    warnings.warn(
        "NOTE: The dataset used to compute Fisher information matrix is loaded "
        "with batch size 1.")
    assert (isinstance(fisher_params_sample_percentage, int) and
            0 <= fisher_params_sample_percentage <= 100)
    fisher_params_ds = load_data(
        dataset_name=fisher_params_dataset,
        scene_type=fisher_params_scene,
        fraction=f"[{fisher_params_sample_percentage}%]",
        batch_size=1,
        shuffle_data=True)
    return train_ds, val_ds, test_ds, fisher_params_ds
