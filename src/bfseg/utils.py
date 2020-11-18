import gdown
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import mark_boundaries
from fast_slic import Slic
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage
from skimage.segmentation import watershed
from skimage.feature import canny


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


"""
Helper functions to convert self supervised sparse annotations to images
"""


def reduce_superpixel(seeds_bg, seeds_fg, assignment):
  """
     seeds_bg: List with points that belong to the background class
     seeds_fg: List with points that belong to the foreground class
     assignment: Superipixel assignment as produced by e.g. SLIC

     Assigns each superpixel in assignment a label [foreground, background, None] based on majority voting of seeds in it
     Resulting assignment will have the following labeling:
     0 -> Background
     1 -> No label
     2 -> Foreground


     Improvement ideas:
        - Use variance of distances to  mesh over superpixel. If variance is too big do not assign any label
        - Use Absolute distance of points, points that are closer to us have higher information value than points that are further away
    """
  # How many superpixels are there
  max_points = np.max(assignment) + 1

  # For each superpixel, store how many foreground - background votes it received
  accounting = np.zeros(max_points)

  for pointX, pointY in seeds_bg:
    # Get Superpixel
    superpixel_label = assignment[pointX, pointY]
    # Increase Vote by 1
    accounting[superpixel_label] = accounting[superpixel_label] + 1

  for pointX, pointY in seeds_fg:
    # Get superpixel
    superpixel_label = assignment[pointX, pointY]
    # Decrease Vote by 1
    accounting[superpixel_label] = accounting[superpixel_label] - 1

  for i in range(max_points):
    assignment[assignment == i] = (np.sign(accounting[i]) + 1)

  return assignment


# Threshold between foreground and background.
# Labels have values [1,255] where 1 means the point is really close to the mesh and 255 means the point is really far away from  a mesh
# a threshold of 100 means all points with distance value > 100 will be assigned to "foreground" label
class_foreground = 0
class_background = 2
class_unknown = 1


def plot_reduced_sp(seeds_bg, seeds_fg, assignment, original):
  """ Plots everything used to generate the labels """
  plt.subplot(3, 3, 1)
  plt.imshow(mark_boundaries(original, assignment))
  assign = reduce_superpixel(seeds_bg, seeds_fg, assignment)
  plt.subplot(3, 3, 2)
  plt.imshow(to_rgb_mask(assign))
  plt.subplot(3, 3, 3)
  plt.imshow(mark_boundaries(original, assign))

  plt_cnt = 0
  for i in (class_background, class_foreground, class_unknown):
    plt.subplot(3, 3, 3 + plt_cnt)
    plt.imshow(assign == i)
    plt_cnt = plt_cnt + 1
  plt.subplot(3, 3, 6)
  plt.imshow(
      np.multiply(
          original,
          np.stack([
              assign == class_background, assign == class_background, assign
              == class_background
          ],
                   axis=-1)))
  plt.title("background")
  plt.subplot(3, 3, 7)
  plt.imshow(
      np.multiply(
          original,
          np.stack([
              assign == class_foreground, assign == class_foreground, assign
              == class_foreground
          ],
                   axis=-1)))
  plt.title("foreground")
  plt.subplot(3, 3, 8)
  plt.imshow(
      np.multiply(
          original,
          np.stack([
              assign == class_unknown, assign == class_unknown, assign
              == class_unknown
          ],
                   axis=-1)))
  plt.title("unknown")


def to_rgb_mask(mask):
  # Converts a labels mask to RGB to better see which pixel is what class
  mask_r = (mask == class_background) * 255
  mask_g = (mask == class_foreground) * 255
  mask_b = (mask == class_unknown) * 255
  return Image.fromarray(np.uint8(np.stack([mask_r, mask_g, mask_b], axis=-1)))


def get_preview_image(original, binary_mask):
  # Projects the labels mask on a rgb image
  img = to_rgb_mask(binary_mask)
  img.putalpha(50)
  new_img = original.copy()
  new_img.paste(img, (0, 0), img)
  return new_img


def convert_file_path_to_gt(label,
                            original,
                            distance,
                            outfolder,
                            resize=True,
                            useSuperpixel=True,
                            onlyPlotResults=False,
                            fg_bg_threshold=100):
  # label: Path to label file as generated by ros node
  # original: Path to original image
  # distance: Path to distance Imag as generated by rose node. TODO use this to refine superpixel aswell as to generate distance labels
  # resize: Wether to resize the images -> Better performance for SLIC
  # userSuperpixel: Wether to user Superixel or watershed algorithm
  #
  # Converts the given sparse labels into a aggregated form and stores it in outfolder

  original = Image.open(original)
  labels = Image.open(label)
  distance = Image.open(distance)

  if resize:
    original = original.resize((original.width // 2, original.height // 2),
                               Image.ANTIALIAS)
    labels = labels.resize((labels.width // 2, labels.height // 2), Image.NONE)
    distance = distance.resize((distance.width // 2, distance.height // 2),
                               Image.NONE)

  np_orig = np.asarray(original)
  np_labels = np.asarray(labels)
  # Foreground Labels
  np_labels_foreground = np_labels > fg_bg_threshold
  # Background Labels
  np_labels_background = np.logical_and(np_labels < fg_bg_threshold,
                                        np_labels > 0)
  # No Labels
  np_no_labels = np_labels == 0

  # Create a list containing all points that were assigned background
  seeds_bg = [
      (c[0], c[1]) for c in np.asarray(np.where(np_labels_background > 0)).T
  ]
  # Create a list containing all points that were assigned foreground
  seeds_fg = [
      (c[0], c[1]) for c in np.asarray(np.where(np_labels_foreground > 0)).T
  ]

  if useSuperpixel:
    # superpixels = slic.iterate(np_orig)
    superpixels = skimage.segmentation.slic(np_orig,
                                            n_segments=1000,
                                            compactness=4,
                                            sigma=1,
                                            start_label=1)

    if onlyPlotResults:
      plot_reduced_sp(seeds_bg, seeds_fg, superpixels, original)
      return

    mask = reduce_superpixel(seeds_bg, seeds_fg, superpixels)
  else:
    markers = np.zeros(np_labels_foreground.shape, dtype=np.uint)
    # assign values to markers. +1 since zero will be ignored as marker assignement
    markers[np_labels_foreground] = class_foreground + 1
    markers[np_labels_background] = class_background + 1

    height, width = np_labels_background.shape
    step = 20
    top_padding = 10
    bot_padding = 10
    for i in range(width // step):
      markers[top_padding, step * i] = class_unknown + 1
      markers[2 * top_padding, step * i] = class_unknown + 1
      markers[height - bot_padding, step * i] = class_unknown + 1

    # Run watershed on canny edge filtered image.
    mask = watershed(
        skimage.feature.canny(skimage.color.rgb2gray(np_orig), sigma=0.1),
        markers) - 1
    if onlyPlotResults:
      plt.subplot(1, 3, 1)
      plt.imshow(original)
      plt.subplot(1, 3, 2)
      plt.imshow(
          skimage.feature.canny(skimage.color.rgb2gray(np_orig), sigma=0.1))
      plt.title("canny edge image")
      plt.subplot(1, 3, 3)
      plt.imshow(to_rgb_mask(mask))
      return

  Image.fromarray(np.uint8(mask), 'L').save(outfolder + "semseg.png")
  get_preview_image(original, mask).save(outfolder + "preview.png")
