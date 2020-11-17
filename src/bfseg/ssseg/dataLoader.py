"""
    Contains all logic that is used to load images from the disk
"""
import tensorflow as tf
import os
import random


class DataLoader:

  def __init__(self,
               workingDir,
               inputSize,
               outputSize=None,
               validationDir=None,
               batchSize=4,
               shuffleBufferSize=64):
    self.workingDir = workingDir
    self.batchSize = batchSize
    self.shuffleBufferSize = shuffleBufferSize
    self.inputSize = inputSize

    if outputSize is None:
      self.outputSize = inputSize
    else:
      self.outputSize = outputSize

    if validationDir is None:
      self.validationDir = workingDir
    else:
      self.validationDir = validationDir

    self.filenames, self.labels = self.getImageDataFromPath(self.workingDir)
    self.validationFiles, self.validationLabels = self.getImageDataFromPath(
        self.validationDir)
    self.size = len(self.filenames)
    self.validationSize = len(self.validationFiles)

  def getImageDataFromPath(self, path):
    """ returns all input images and labels stored in the given 'path' folder.
        The folder structure looks like this:
        path/
            img000/
             - img.[png/jpg]   <- Original image
             - semseg.png      <- Labels for Semantic Segmentation
             - distance.png    <- Labels for depth prediction
             - ... additional information e.g. pc, pose
            img999/
              ....

        Currently the distance label is unused
        """
    # lists to return
    labels = []
    imgs = []

    # iterate through all imgXXX folders
    for image_folders in sorted(os.listdir(path)):
      # make sure it is folder
      if os.path.isdir(os.path.join(path, image_folders)):
        # cache folder content (e.g. img.png, semseg.png)
        folder_content = sorted(os.listdir(os.path.join(path, image_folders)))
        # count how many semseg images (=labels) are there
        semantic_labels = [
            fileName for fileName in folder_content if "semseg" in fileName
        ]
        # count how many original images are there
        images = [fileName for fileName in folder_content if "img" in fileName]

        if len(semantic_labels) == len(images):
          imgs.extend(images)
          labels.extend(labels)
        else:
          print("WARNING! Label / Image missmatch in folder:",
                path + "image_folders")

    return imgs, labels

  def parse_function(self, filename, label):
    """ Read image and labels. Will crash if filetype is neither jpg nor png. """
    if tf.io.is_jpeg(filename):
      image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
    else:
      image = tf.image.decode_png(tf.io.read_file(filename), channels=3)
    labels = tf.image.decode_png(tf.io.read_file(label), channels=1)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Make sure to NOT mess up the labels thus use nearest neighbour interpolation
    return self.cropImageToInputSize(
        image, self.inputSize), self.cropImageToInputSize(labels,
                                                          self.outputSize,
                                                          method="nearest")

  def cropImageToInputSize(self, image, size, method="bilinear"):
    """ Crop image. Removes a little bit from the top of the image, as any won't have labels for this area """

    # TODO do not hardcode, make image size dependant
    # image_w, image_h, _ = tf.unstack(tf.shape(image))
    # image_w = int(image_w)
    # image_h = int(image_h)
    # # Remove top 10 and bottom 10%
    # cut_top = int(image_h * 0.1)
    # crop_height = int((1 - 0.2) * image_h)
    #
    # # Calculate width of cropped image to have the right shape
    # rel_scale = crop_height / size[0]
    # crop_width = int(image_w * rel_scale)
    # cut_left = int((image_w - crop_width) / 2)

    image = tf.image.resize(image, [600, 900], method=method)

    cropped_image = tf.image.crop_to_bounding_box(image, 100, 0, 480, 640)
    # Resize it to desired input size of the network
    return tf.image.resize(cropped_image, size, method=method)

  def train_preprocess(self, image, label):
    """
         Args:
             image: keras tensor containing rgb image
             label: keras tensor containing label image (1 channel)

         Returns: randomly augmented image

         """

    # random data augmentation can be uncommented here

    # # Flip left right
    # image = tf.image.random_flip_left_right(image)
    # # Change brightness
    # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # # Change saturation
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

  def getDataset(self):
    """
        Returns two tf.data.Datasets.
        First one contains images width self supervised labels generated from the pointcloud
        Second one contains images width ground truth annotations
        """

    return self.getDatasetForList(self.filenames,
                                  self.labels), \
           self.getDatasetForList(
               self.validationFiles,
               self.validationLabels)

  def getDatasetForList(self, imgs, labels):
    """ Returns a tensorflow dataset based on list of filenames """
    return tf.data.Dataset.from_tensor_slices((imgs, labels)) \
        .shuffle(len(imgs)) \
        .map(self.parse_function, num_parallel_calls=4) \
        .map(self.train_preprocess, num_parallel_calls=4) \
        .cache() \
        .shuffle(self.shuffleBufferSize) \
        .batch(4) \
        .prefetch(tf.data.experimental.AUTOTUNE)
