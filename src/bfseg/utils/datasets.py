""" Utils to pre-process and load the datasets.
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import warnings
import cv2 

from bfseg.data.fsdata import load_fsdata
from bfseg.utils.images import augmentation_with_mask, augmentation_with_mask_depth, preprocess_median_full_with_mask_depth, preprocess_median_full_with_mask
from bfseg.utils.replay_buffer import ReplayBuffer
import scipy.stats as stats 


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
  labels (0, 1, 2) where all classes that belong to the background (e.g., floor,
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


@tf.function
def preprocess_bagfile(image, label):
  r"""Preprocesses bagfile dataset (e.g., Rumlang). The dataset consists of
  three labels (0, 1, 2) with the following meaning:
  - 0: foreground
  - 1: background
  - 2: unsure (ignored in training)
  """
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize image and label.
  image = tf.image.resize(image, (480, 640),
                          method=tf.image.ResizeMethod.BILINEAR)
  label = tf.image.resize(label, (480, 640),
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  # Mask out unknown pixels.
  mask = tf.squeeze(tf.not_equal(label, 2))
  label = tf.cast(label == 1, tf.uint8)
  image = tf.cast(image, tf.float32)
  return image, label, mask

@tf.function 
def preprocess_bagfile_depth(image, label):
  r"""Preprocesses bagfile dataset (e.g., Rumlang). The dataset consists of
  three labels (0, 1, 2) with the following meaning:
  - 0: foreground
  - 1: background
  - 2: unsure (ignored in training)
  """
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize image and label.
  image = tf.image.resize(image, (480, 640),
                          method=tf.image.ResizeMethod.BILINEAR)
  seg_label = tf.image.resize(label['seg'], (480, 640),
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  depth_label = tf.image.resize(label['distance'], (480, 640),
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # Convert depth [0,255] to real distance [0m, 10m] (TODO: remove hardcoded 10)
  depth_norm = ((tf.cast(depth_label, dtype=tf.float32) - 1.0) * 10 / 254)
  mode = "boxcox_standardize"
  if mode == "boxcox_standardize":
    print("boxcox standardize")
    depth_norm_2 = tf.where(
        tf.equal(depth_label, tf.constant(0, dtype=tf.uint8)),
        tf.constant(float('nan'), dtype=tf.float32), depth_norm)

    old_shape = depth_norm_2.shape
    depth_flattened = tf.reshape(depth_norm_2, [-1])
    depth_flattened_trans, _  = tf_boxcox(depth_flattened)
    depth_restored = tf.reshape(depth_flattened_trans, old_shape)    
    depth_standardized = tf.image.per_image_standardization(depth_restored)
    depth_label_final = depth_standardized
  
  elif mode == "inverse_standardize":
    print("inverse standardize")
    depth_norm_2 = tf.where(
          tf.equal(depth_label, tf.constant(0, dtype=tf.float32)),
          tf.constant(float('nan'), dtype=tf.float32), depth_norm)

    depth_inverse = 10 / depth_norm_2  
    depth_inverse_standardized = tf.image.per_image_standardization(depth_inverse)
    depth_label_final = depth_inverse_standardized

  else: 
    depth_norm_2 = tf.where(
            tf.equal(depth_label, tf.constant(0, dtype=tf.uint8)),
            tf.constant(float('nan'), dtype=tf.float32), depth_norm)
      # normalize depth (inverse)
    depth_label_final = 10 / depth_norm_2

  # Semseg: Mask out unknown pixels for semseg
  seg_mask = tf.squeeze(tf.not_equal(seg_label, 2))
  seg_label = tf.cast(seg_label == 1, tf.uint8)

  # Depth mask for MAE only (not used in loss)
  depth_mask = tf.squeeze(tf.not_equal(depth_norm, 0))

  image = tf.cast(image, tf.float32)
  labels = {'seg_label': seg_label, 'depth_label': depth_label_final}
  masks = {'seg_mask': seg_mask, 'depth_mask': depth_mask}
  return image, labels, masks

@tf.function
def preprocess_nyu_depth(image, label):
  r"""Preprocesses NYU dataset:
  - Normalize images: `uint8` -> `float32`.
  - Assigns label: 1 if belong to background, 0 if foreground.
  - Creates all-True mask, since NYU labels are all known.
  """
  seg_label = label['seg']
  depth_label = tf.expand_dims(label['distance'], -1)
  seg_mask = tf.not_equal(seg_label, -1)  # All true.
  seg_label = tf.expand_dims(seg_label, axis=2)
  image = tf.cast(image, tf.float32) / 255.
  
  mode = "inverse_standardize"
  if mode == "boxcox_standardize":
    print("boxcox standardize")
    depth_norm_2 = tf.where(
          tf.equal(depth_label, tf.constant(0, dtype=tf.float32)),
          tf.constant(float('nan'), dtype=tf.float32), depth_label)

    old_shape = depth_norm_2.shape
    depth_flattened = tf.reshape(depth_norm_2, [-1])
    depth_flattened_trans, _  = tf_boxcox(depth_flattened)
    depth_restored = tf.reshape(depth_flattened_trans, old_shape)    
    depth_standardized = tf.image.per_image_standardization(depth_restored)
    depth_label_final = depth_standardized
  
  elif mode == "inverse_standardize":
    print("inverse standardize")
    depth_norm_2 = tf.where(
          tf.equal(depth_label, tf.constant(0, dtype=tf.float32)),
          tf.constant(float('nan'), dtype=tf.float32), depth_label)

    depth_inverse = 10 / depth_norm_2  
    depth_inverse_standardized = tf.image.per_image_standardization(depth_inverse)
    depth_label_final = depth_inverse_standardized

  else: 
    # clip max depth value to 10
    depth_label = tf.where(
          tf.math.greater_equal(depth_label, tf.constant(10, dtype=tf.float32)),
          tf.constant(float(10), dtype=tf.float32), depth_label)

    # replace zeros with NaN for depth
    depth_norm_2 = tf.where(
          tf.equal(depth_label, tf.constant(0, dtype=tf.float32)),
          tf.constant(float('nan'), dtype=tf.float32), depth_label)
    # normalize depth (inverse)
    depth_label_final = 10 / depth_norm_2

  # Depth mask for MAE only (not used in loss)
  depth_mask = tf.squeeze(tf.not_equal(depth_label, 0))

  labels = {'seg_label': seg_label, 'depth_label': depth_label_final}
  masks = {'seg_mask': seg_mask, 'depth_mask': depth_mask}
  return image, labels, masks



  


@tf.function
def preprocess_bagfile_different_dataloader(blob):
  r"""Overload of `preprocess_bagfile` for different data loader
  (`bfseg.data.fsdata.load_fsdata`).
  """
  return preprocess_bagfile(image=blob['rgb'],
                            label=tf.expand_dims(blob['labels'], axis=-1))


@tf.function
def preprocess_hive(image, label):
  r"""Preprocesses the manually annotated datasets, by just making sure that
  the RGB values are between 0 and 1.
  """
  image = tf.image.convert_image_dtype(image, tf.float32)

  return image, label


def load_data(dataset_name, scene_type, fraction, batch_size, shuffle_data):
  r"""Creates a data loader given the dataset parameters as input.
  Args:
    dataset_name (str): Name of the dataset. Valid entries are:
      "NyuDepthV2Labeled" (NYU dataset), "BfsegCLAMeshdistLabels",
      "MeshdistPseudolabels" (bagfile dataset, i.e., Rumlang/garage/office),
      "BfsegValidationLabeled" (CLA validation dataset, with manually-annotated
      labels), "OfficeRumlangValidationLabeled" (bagfile validation dataset,
      i.e., Rumlang/office, with manually-annotated labels).
    scene_type (str): Type of scene in the dataset to use type. Valid entries
      are:
      - If `dataset_name` is "NyuDepthV2Labeled":
        - None: All the scenes in the dataset are selected.
        - "kitchen": Kitchen scene only.
        - "bedroom": Bedroom scene only.
      - If `dataset_name` is "BfsegCLAMeshdistLabels":
        - None: All the samples in the dataset are selected (no scene
            subdivision is available).
      - If `dataset_name` is "MeshdistPseudolabels":
        - None: All the scenes in the dataset are selected.
        - "garage_full": All the three scenes from the garage.
        - "rumlang_full": Both the scenes from Rumlang.
        - One of the two following scenes:
          - "garage1"
          - "garage2"
          - "garage3"
          - "office4"
          - "office4_2302"
          - "office4_2402"
          - "office5"
          - "office6_2502"
          - "rumlang2"
          - "rumlang3"
      - If `dataset_name` is "BfsegValidationLabeled":
        - None: All the scenes in the dataset are selected.
        - "CLA"
        - "ARCHE"
      - If `dataset_name` is "OfficeRumlangValidationLabeled":
        - None: All the scenes in the dataset are selected.
        - "OFFICE"
        - "RUMLANG"
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
  elif (dataset_name == 'NyuDepthV2LabeledDepth'):
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
  elif (dataset_name == 'MeshdistPseudolabels'):
    if (scene_type is None):
      name = 'garage1+garage2+garage3+rumlang2+rumlang3'
    elif (scene_type == "garage_full"):
      name = "garage1+garage2+garage3"
    elif (scene_type == "rumlang_full"):
      name = "rumlang2+rumlang3"
    elif (scene_type == "office4_2302"):
      ds = load_fsdata(
          '/cluster/work/riner/users/blumh/pickelhaube_full_office4')
      name = scene_type
    elif (scene_type == "office4_2402"):
      ds = load_fsdata('/cluster/work/riner/users/blumh/'
                       'pickelhaube_full_office4_agreement_dump')
      name = scene_type
    elif (scene_type == "office6_2502"):
      ds = load_fsdata(
          '/cluster/work/riner/users/blumh/pickelhaube_full_office6')
      name = scene_type
    elif (scene_type in [
        "garage1", "garage2", "garage3", "office4", "office5", "rumlang2",
        "rumlang3"
    ]):
      name = scene_type
  elif (dataset_name == 'MeshdistPseudolabelsDyn'):
    if (scene_type == "rumlang1_full_dyn"):
      name = "rumlang1_full_dyn"
    elif (scene_type == "rumlang1_full"):
      name = "rumlang1_full"
    elif (scene_type == "rumlang1_full_test1"):
      name = "rumlang1_full_test1"
    else:
      raise Exception("Invalid scene type: %s!" % scene_type)
  elif (dataset_name == 'MeshdistPseudolabelsDense'):
    if (scene_type in ['office6_sparse50_allcams', 'office6_paper', 'office12_dense20_dyn_cam2', 'office12_sparse50_dyn_cam2', 
      'office12_sparse50_dyn_allcams', 'office3_dense20_dyn_cam2', 'office3_sparse50_dyn_cam2', 'office3_sparse50_dyn_allcams']):
      name = scene_type
    else:
      raise Exception("Invalid scene type: %s!" % scene_type)
  elif (dataset_name == 'MeshdistPseudolabelsDenseDepth'):
    if (scene_type in ["office3_densedepth20_dyn_test", "office3_densedepth20_dyn_cam2", "office12_densedepth20_dyn_cam2"]):
      name = scene_type
    else:
      raise Exception("Invalid scene type: %s!" % scene_type)
  elif (dataset_name == 'BfsegValidationLabeled'):
    if (scene_type is None):
      name = 'CLA+ARCHE'
    elif (scene_type in ["CLA", "ARCHE"]):
      name = scene_type
    else:
      raise Exception("Invalid scene type: %s!" % scene_type)
  elif (dataset_name == 'OfficeRumlangValidationLabeled'):
    if (scene_type is None):
      name = 'OFFICE+RUMLANG'
    elif (scene_type in ["OFFICE", "RUMLANG"]):
      name = scene_type
    else:
      raise Exception("Invalid scene type: %s!" % scene_type)
  else:
    raise Exception("Dataset %s not found!" % dataset_name)

  # Select the fraction of samples from the scene.
  if (fraction is not None):
    if (scene_type in ["office4_2302", "office4_2402", "office6_2502"]):
      dataset_length = sum(1 for _ in ds)
      split_fraction = fraction.split(":")
      assert (len(split_fraction) == 2)
      if (split_fraction[0] == "["):
        assert (split_fraction[1][-2:] == "%]")
        fraction = split_fraction[1][:-2]
        num_samples_to_take = int(dataset_length * int(fraction) / 100.)
        ds = ds.take(num_samples_to_take)
        dataset_length = num_samples_to_take
      else:
        assert (split_fraction[0][0] == "[" and split_fraction[0][-1] == "%")
        assert (split_fraction[1] == "]")
        fraction = split_fraction[0][1:-1]
        num_samples_to_skip = int(dataset_length * int(fraction) / 100.)
        ds = ds.skip(num_samples_to_skip)
        dataset_length -= num_samples_to_skip
    else:
      # Handle the special case of a mix of scenes.
      scenes_unmixed = name.split('+')
      if (len(scenes_unmixed) > 1):
        assert ("office4_2302" not in scenes_unmixed and
                "office4_2402" not in scenes_unmixed and
                "office6_2502" not in scenes_unmixed)
        split = f"{scenes_unmixed[0]}{fraction}"
        for scene_unmixed in scenes_unmixed[1:]:
          split += f"+{scene_unmixed}{fraction}"
      else:
        split = f"{name}{fraction}"
  else:
    split = name

  # Actually load the dataset.
  if (scene_type not in ["office4_2302", "office4_2402", "office6_2502"]):
    ds = tfds.load(dataset_name,
                   split=split,
                   shuffle_files=shuffle_data,
                   as_supervised=True)
    dataset_length = len(ds)
  # Apply pre-processing.
  if (dataset_name == 'NyuDepthV2Labeled'):
    ds = ds.map(preprocess_nyu,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif (dataset_name == 'NyuDepthV2LabeledDepth'):
    ds = ds.map(preprocess_nyu_depth,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif (dataset_name == 'BfsegCLAMeshdistLabels'):
    ds = ds.map(preprocess_cla,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif (dataset_name == 'MeshdistPseudolabels'):
    if (scene_type in ["office4_2302", "office4_2402", "office6_2502"]):
      ds = ds.map(preprocess_bagfile_different_dataloader,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
      ds = ds.map(preprocess_bagfile,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif (dataset_name == 'MeshdistPseudolabelsDyn'):
    ds = ds.map(preprocess_bagfile,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif (dataset_name == 'MeshdistPseudolabelsDense'):
    ds = ds.map(preprocess_bagfile,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif (dataset_name == 'MeshdistPseudolabelsDenseDepth'):
    ds = ds.map(preprocess_bagfile_depth,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif (dataset_name
        in ['BfsegValidationLabeled', 'OfficeRumlangValidationLabeled']):
    ds = ds.map(preprocess_hive,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.cache()
  # Further shuffle.
  if (shuffle_data):
    ds = ds.shuffle(dataset_length)

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
  r"""Creates data loaders, for training, validation and testing, and optionally
  for the dataset required to compute Fisher parameters.
  """
  assert (isinstance(validation_percentage, int) and
          0 <= validation_percentage <= 100)
  assert ((fisher_params_dataset is None) == (fisher_params_sample_percentage is
                                              None))
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
  if (test_dataset is not None):
    test_ds = load_data(dataset_name=test_dataset,
                        scene_type=test_scene,
                        fraction=None,
                        batch_size=batch_size,
                        shuffle_data=False)
  else:
    test_ds = None
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
        fraction=f"[:{fisher_params_sample_percentage}%]",
        batch_size=1,
        shuffle_data=True)
    return train_ds, val_ds, test_ds, fisher_params_ds


def load_replay_datasets(replay_datasets, replay_datasets_scene, batch_size):
  r"""Creates data loaders for one or more replay datasets.
  """
  assert (isinstance(replay_datasets, list) and
          isinstance(replay_datasets_scene, list) and
          len(replay_datasets) == len(replay_datasets_scene))
  replay_ds = {}
  for curr_replay_dataset, curr_replay_dataset_scene in zip(
      replay_datasets, replay_datasets_scene):
    replay_ds[f"{curr_replay_dataset}_{curr_replay_dataset_scene}"] = load_data(
        dataset_name=curr_replay_dataset,
        scene_type=curr_replay_dataset_scene,
        fraction=None,
        batch_size=batch_size,
        shuffle_data=False)

  return replay_ds


def update_datasets_with_replay_and_augmentation(
    train_no_replay_ds, test_ds, fraction_replay_ds_to_use,
    ratio_main_ds_replay_ds, replay_datasets, replay_datasets_scene, batch_size,
    perform_data_augmentation, contains_depth=False, perform_preprocessing=False):
  r"""Returns training and test datasets after creating a replay buffer and
  performing data augmentation, if necessary.
  Args:
    train_no_replay_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset):
      Training dataset before replay and augmentation.
    test_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Test
      dataset before replay and augmentation.
    fraction_replay_ds_to_use (float): Cf. `src/bfseg/utils/replay_buffer.py`.
    ratio_main_ds_replay_ds (list of int): Cf.
      `src/bfseg/utils/replay_buffer.py`.
    replay_datasets (list of str). Cf. `load_replay_datasets`.
    replay_datasets_scene (list of str). Cf. `load_replay_datasets`.
    batch_size (int): Batch size to use in the optional replay buffer.
    perform_data_augmentation (bool): Whether or not to perform data
      augmentation.
  Returns:
    train_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Training
      dataset after optional replay and augmentation.
    test (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Test dataset
      after optional replay and augmentation.
  """
  if (fraction_replay_ds_to_use is not None or
      ratio_main_ds_replay_ds is not None):
    # Load replay datasets.
    replay_ds = load_replay_datasets(
        replay_datasets=replay_datasets,
        replay_datasets_scene=replay_datasets_scene,
        batch_size=batch_size)
    replay_buffer = ReplayBuffer(
        main_ds=train_no_replay_ds,
        replay_ds=list(replay_ds.values()),
        batch_size=batch_size,
        ratio_main_ds_replay_ds=ratio_main_ds_replay_ds,
        fraction_replay_ds_to_use=fraction_replay_ds_to_use,
        perform_data_augmentation=perform_data_augmentation,
        contains_depth=contains_depth,
        perform_preprocessing=perform_preprocessing)
    train_ds = replay_buffer.flow()
    # When using replay, evaluate separate metrics only on training set without
    # replay.
    test_ds = {'test': test_ds, 'train_no_replay': train_no_replay_ds}
    # Add replay datasets to "test", i.e., test the performance of the trained
    # model on them.
    # - Ensure that there is no name collision.
    assert (len(set(test_ds.keys()).intersection(replay_ds.keys())) == 0)
    test_ds.update(replay_ds)
  else:
    train_ds = train_no_replay_ds
    # Check if data augmentation should be used.
    if (perform_data_augmentation):
      # Special augmentation function in case augmentation is with depth (TODO: improve)
      if (contains_depth):
        train_ds = train_ds.map(augmentation_with_mask_depth)
      else:
        train_ds = train_ds.map(augmentation_with_mask)
    if (perform_preprocessing):
      if (contains_depth):
        train_ds = train_ds.map(preprocess_median_full_with_mask_depth)
      else:
        train_ds = train_ds.map(preprocess_median_full_with_mask)

  return train_ds, test_ds


def tf_boxcox(input):
  y = tf.numpy_function(stats.boxcox, [input], (tf.float32, tf.float64))
  return y 
