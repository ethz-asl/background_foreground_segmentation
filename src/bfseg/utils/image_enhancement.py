# %load ../src/bfseg/utils/image_enhancement.py
from PIL import Image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.segmentation import watershed
from scipy.spatial import distance
"""
Helper functions to convert self supervised sparse annotations to images
"""


def closest_point_distance(point, points):
  return distance.cdist([point], points).min()


def reduce_superpixel(seeds_bg,
                      seeds_fg,
                      assignment,
                      distance,
                      stdDevThreshold=0.5):
  """
         seeds_bg: List with points that belong to the background class
         seeds_fg: List with points that belong to the foreground class
         assignment: Superipixel assignment as produced by e.g. SLIC
         distance: PNG containing distance information for each lidar point
         stdDevThreshold: Threshold. If variance of superpixel is bigger than this threshold, discard this superpixel

         Assigns each superpixel in assignment a label [foreground, background, None] based on majority voting of seeds in it
         Resulting assignment will have the following labeling:
         0 -> Background
         1 -> No label
         2 -> Foreground


         Improvement ideas:
            - Use Absolute distance of points, points that are closer to us have higher information value than points that are further away
        """
  # How many superpixels are there
  num_points = np.max(assignment) + 1

  # For each superpixel, store how many foreground - background votes it received
  accounting = np.zeros(num_points)
  # Store standard deviation of pixels inside superpixel
  distSquared = np.zeros(num_points)
  distSum = np.zeros(num_points)
  distCounter = np.zeros(num_points)
  for seed, vote in [(seeds_bg, 1), (seeds_fg, -1)]:
    for pointX, pointY in seed:
      # Get Superpixel
      superpixel_label = assignment[pointX, pointY]
      # Increase Vote by 1
      accounting[superpixel_label] = accounting[superpixel_label] + vote
      # Get Distance of this point
      dist = distance[pointX, pointY] * 10 / 255
      distCounter[superpixel_label] = distCounter[superpixel_label] + 1
      distSum[superpixel_label] += dist
      distSquared[superpixel_label] += dist * dist

  mean = distSum / distCounter
  stdDev = (distSquared / distCounter) - (mean * mean)
  stdDev[np.isnan(stdDev)] = 0

  for i in range(num_points):
    label = np.sign(accounting[i]) + 1
    if stdDev[i] > stdDevThreshold:
      label = 1
    assignment[assignment == i] = label

  return assignment


class_foreground = 0
class_background = 2
class_unknown = 1


def plot_reduced_sp(seeds_bg, seeds_fg, assignment, distance, original):
  """ Plots everything used to generate the labels """
  plt.subplot(3, 3, 1)
  plt.imshow(mark_boundaries(original, assignment))
  assign = reduce_superpixel(seeds_bg, seeds_fg, assignment, distance)
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
                            foregroundTrustRegion=True,
                            onlyPlotResults=False,
                            superpixels=None,
                            fg_bg_threshold=100):
  """
    label: Path to label file as generated by ros node
    original: Path to original image
    distance: Path to distance Imag as generated by rose node. TODO use this to refine superpixel aswell as to generate distance labels
    outfolder: Path to output folder
    resize: Wether to resize the images -> Better performance for SLIC
    userSuperpixel: Wether to user Superixel or watershed algorithm
    foregroundTrustRegion: Flag, if enabled, foreground points have a "circle around it called trust region" Background points inside this circle will be removed.
    onlyPlotResults: Flagm if enabled only plots results and does not store them on disk
    superpixels: Pillow image, containing superpixel assignement or NONE -> use slic
    fg_bg_threshold: Threshold [cm] points thare are further away from the mesh than this threshold will be treated as foreground

    Converts the given sparse labels into a aggregated form and stores it in outfolder """

  original = Image.open(original)
  labels = Image.open(label)
  distance = Image.open(distance)

  if resize:
    original = original.resize((original.width // 2, original.height // 2),
                               Image.ANTIALIAS)
    labels = labels.resize((labels.width // 2, labels.height // 2), Image.NONE)
    distance = distance.resize((distance.width // 2, distance.height // 2),
                               Image.NONE)

    if superpixels is not None:
      superpixels = superpixels.resize(
          (superpixels.width // 2, superpixels.height // 2), Image.NONE)
  if superpixels is not None:
    superpixels = np.asarray(superpixels).copy()
  distance = np.asarray(distance)
  np_orig = np.asarray(original)
  np_labels = np.asarray(labels)
  # Foreground Labels
  np_labels_foreground = np_labels > fg_bg_threshold
  # Background Labels
  np_labels_background = np.logical_and(np_labels < fg_bg_threshold,
                                        np_labels > 0)
  # No Labels
  np_no_labels = np_labels == 0

  seeds_fg_arr = np.asarray(
      [[c[0], c[1]] for c in np.asarray(np.where(np_labels_foreground > 0)).T])

  # Create a list containing all points that were assigned background
  seeds_bg = [(c[0], c[1])
              for c in np.asarray(np.where(np_labels_background > 0)).T
              if not (foregroundTrustRegion and closest_point_distance(
                  np.asarray([c[0], c[1]]), seeds_fg_arr) < 20)]
  # Create a list containing all points that were assigned foreground
  seeds_fg = [
      (c[0], c[1]) for c in np.asarray(np.where(np_labels_foreground > 0)).T
  ]

  if useSuperpixel:
    if superpixels is None:
      # If no superpixel image is probided, use slic
      superpixels = skimage.segmentation.slic(np_orig,
                                              n_segments=1000,
                                              compactness=4,
                                              sigma=1,
                                              start_label=1)

    if onlyPlotResults:
      plot_reduced_sp(seeds_bg, seeds_fg, superpixels, distance, original)
      return

    mask = reduce_superpixel(seeds_bg, seeds_fg, superpixels, distance)
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

  timeStamp = outfolder.split("/")[-2]
  Image.fromarray(np.uint8(mask),
                  'L').save(outfolder + timeStamp + "_semseg.png")
  original.save(outfolder + timeStamp + "_img.png")
