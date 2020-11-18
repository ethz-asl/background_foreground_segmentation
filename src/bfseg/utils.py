import gdown
from os import path
from PIL import Image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.segmentation import watershed


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


def load_gdrive_file(file_id, ending=''):
  """Downloads files from google drive, caches files that are already downloaded."""
  filename = '{}.{}'.format(file_id, ending) if ending else file_id
  filename = path.join(path.expanduser('~/.keras/datasets'), filename)
  if not path.exists(filename):
    gdown.download('https://drive.google.com/uc?id={}'.format(file_id),
                   filename,
                   quiet=False)
  return filename


