import argparse
import gdown
import h5py
import numpy as np
from os import path
import re
import tensorflow as tf
import tensorflow_datasets as tfds


@tf.function
def normalize_img(image, label):
  """Normalizes images to [0, 1]: `uint8` -> `float32`."""
  label = tf.expand_dims(label, axis=2)
  image = tf.cast(image, tf.float32) / 255.
  return image, label


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def crop_multiple(data, multiple_of=16):
  """Force the array dimension to be multiple of the given factor.
    Args:
        data: a >=2-dim array, first 2 dims will be cropped
        multiple_of: the factor, as an int
    Returns:
        cropped array
    """
  try:
    h, w = data.shape[0], data.shape[1]
  except (AttributeError, IndexError):
    # the data is not in image format, I cannot crop
    return data
  h_c, w_c = [d - (d % multiple_of) for d in [h, w]]
  if h_c != h or w_c != w:
    return data[:h_c, :w_c, ...]
  else:
    return data


def crop_map(*data):
  if len(data) == 1 and isinstance(data, dict):
    for key in data:
      data[0][key] = crop_multiple(data[key])
  elif len(data) == 1 and isinstance(data, list):
    for i in range(len(data)):
      data[0][i] = crop_multiple(data[i])
  elif len(data) > 1:
    data = list(data)
    for i in range(len(data)):
      data[i] = crop_multiple(data[i])
    return (*data,)
  else:
    data = crop_multiple(data)
  return data


def load_gdrive_file(file_id,
                     ending='',
                     output_folder=path.expanduser('~/.keras/datasets')):
  """Downloads files from google drive, caches files that are already downloaded."""
  filename = '{}.{}'.format(file_id, ending) if ending else file_id
  filename = path.join(output_folder, filename)
  if not path.exists(filename):
    gdown.download('https://drive.google.com/uc?id={}'.format(file_id),
                   filename,
                   quiet=False)
  return filename


def dump_meshdist_ds_to_h5(datasets, dump_depth=False, path="data.h5"):
  """
    Dumps all images and dataset information into a .h5 file

    Args:
        datasets: List containing triplets: ("split_name","tf.dataset", [filenames])
        path: Path where h5 file should be stored
    """
  with h5py.File(path, 'w') as hf:
    for (name, ds, file_names) in datasets:
      num_images = len(ds)
      # images are stored as float [0,1]
      images = np.zeros((num_images, 480, 640, 3), dtype=float)
      labels = np.zeros((num_images, 480, 640, 1), dtype=np.uint8)
      if dump_depth:
        depth = np.zeros((num_images, 480, 640, 1), dtype=float)

      for idx, (image, label) in enumerate(ds):
        print(f" processing img {idx} of {len(ds)}", end="\r")
        images[idx, ...] = image.numpy()
        labels[idx, ...] = label.numpy()
        if dump_depth:
          depth[idx, ...] = depth.numpy()

      grp = hf.require_group(name)
      # export images and labels
      dataset_img = grp.create_dataset("images", np.shape(images), data=images)
      dataset_label = grp.create_dataset("labels",
                                         np.shape(labels),
                                         data=labels)
      if dump_depth:
        dataset_depth = grp.create_dataset("depth", np.shape(depth), data=depth)

      # Now store metadata (camera and timestamp for each image)
      metadata = hf.require_group("metadata").require_group(name)
      # find the string that matches "camX_timestamp"
      # looks like [cam0_1230512.214124, cam1_1234561234.21, ... ]
      info = [
          re.findall("(cam\d)_(\d+\.\d+)",
                     p.split('/')[-1])[0] for p in file_names
      ]
      for idx, (cam, ts) in enumerate(info):
        # e.g. 0, (cam0, 1230512.214124)
        grp = metadata.require_group(str(idx))
        grp.create_dataset(cam, shape=(1, 1), dtype="float64", data=float(ts))


def load_data(dataset_name, step, batch_size, use_pretrain_dataset=False):
  """Loads the training, validation, test datasets and optionally the dataset
  for pretraining. The samples in the datasets are normalized to [0, 1].

  Args:
    dataset_name (str): Name of the dataset to load. This should be a known name
      for tensorflow_datasets.
    step (str): Either "step1" or "step2".
      - If "step1": The splits are the following:
          - Train: The first 80% samples of the 'train' split of the dataset.
          - Validation: The last 20% samples of the 'train' split of the
              dataset.
          - Test: The first 80% samples of the 'test' split of the dataset.
          - Pre-train: None.
      - If "step2": The splits are the following:
          #TODO(fmilano): Check.
          - Train: The first 80% samples of the 'test' split of the dataset.
          - Validation: The last 20% samples of the 'test' split of the dataset.
          - Test: The last 20% samples of the 'train' split of the dataset.
          - Pre-train (optional): The first 80% samples of the 'train' split of
              the dataset.
    batch_size (int): Batch size to use.
    use_pretrain_dataset (bool, default=False): Whether or not to return a
      'pre-train' dataset.

  Returns:
    lr (float): Learning rate. #TODO(fmilano): Change this, learning rate should
      not come from this function.
    train_ds (tf.data.Dataset): Training dataset.
    val_ds (tf.data.Dataset): Validation dataset.
    test_ds (tf.data.Dataset): Test dataset.
    pretrain_ds (tf.data.Dataset, optional): Pre-train dataset. NOTE: this is
      only returned if the :arg:`use_pretrain_dataset` is `True`.
  """
  assert (step in ["step1", "step2"],
          "The input argument `step` must be one of: 'step1', 'step2'.")
  # Load data.
  if (step == "step1"):
    train_ds, train_info = tfds.load(
        dataset_name,
        split='train[:80%]',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    #TODO(fmilano): Make train-val split a parameter.
    val_ds, _ = tfds.load(
        dataset_name,
        split='train[80%:]',
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    #TODO(fmilano): Check why only 80% of the test dataset is used.
    test_ds, _ = tfds.load(
        dataset_name,
        split='test[80%:]',
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    #TODO(fmilano): Make learning rate a parameter.
    lr = 1e-4
  else:
    #TODO(fmilano): Check below.
    train_ds, train_info = tfds.load(
        dataset_name,
        split='test[:80%]',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    val_ds, _ = tfds.load(
        dataset_name,
        split='test[80%:]',
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    test_ds, _ = tfds.load(
        dataset_name,
        split='train[80%:]',
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )

    if (use_pretrain_dataset):
      pretrain_ds, _ = tfds.load(
          dataset_name,
          split='train[:80%]',
          shuffle_files=False,
          as_supervised=True,
          with_info=True,
      )
    #TODO(fmilano): Make learning rate a parameter.
    lr = 1e-5

  # Prepare training dataset.
  # - Perform image normalization.
  train_ds = train_ds.map(normalize_img,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_ds = train_ds.cache()
  # - Shuffle data.
  #TODO(fmilano): Check if needed, `shuffle_files` is already set to True.
  #TODO(fmilano): Check below.
  if (step == "step1"):
    train_ds = train_ds.shuffle(
        int(train_info.splits['train'].num_examples * 0.8))
  else:
    train_ds = train_ds.shuffle(
        int(train_info.splits['test'].num_examples * 0.8))
  # - Prepare batches.
  train_ds = train_ds.batch(batch_size)
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

  # Prepare validation dataset.
  # - Perform image normalization.
  val_ds = val_ds.map(normalize_img,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # - Prepare batches.
  val_ds = val_ds.cache().batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)

  # Prepare test dataset.
  # - Perform image normalization.
  test_ds = test_ds.map(normalize_img,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # - Prepare batches.
  test_ds = test_ds.cache().batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)

  if (use_pretrain_dataset):
    # Prepare pre-train dataset.
    # - Perform image normalization.
    pretrain_ds = pretrain_ds.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # - Prepare batches.
    pretrain_ds = pretrain_ds.cache().batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)

    return lr, train_ds, val_ds, test_ds, pretrain_ds
  else:
    return lr, train_ds, val_ds, test_ds
