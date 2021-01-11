from os import path
import argparse
import re
import h5py
import numpy as np
import gdown


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
