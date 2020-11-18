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
            img_0000/
             - img_0000__img.[png/jpg]   <- Original image
             - img_0000_semseg.png      <- Labels for Semantic Segmentation
             - img_0000distance.png    <- Labels for depth prediction
             - ... additional information e.g. pc, pose
            img_9999/
              ....

        Currently the distance label is unused
        """
    # lists to return
    labels = []
    imgs = []

    # iterate through all imgXXX folders
    for image_folder in sorted(os.listdir(path)):
      image_folder_path = os.path.join(path, image_folder)
      # make sure it is folder
      if os.path.isdir(image_folder_path):
        # cache folder content (e.g. img.png, semseg.png)
        folder_content = sorted(os.listdir(image_folder_path))
        #print(folder_content)
        # count how many semseg images (=labels) are there
        semantic_labels = [
            os.path.join(image_folder_path, fileName)
            for fileName in folder_content
            if image_folder + "_semseg" in fileName
        ]
        # count how many original images are there
        images = [
            os.path.join(image_folder_path, fileName)
            for fileName in folder_content
            if image_folder + "_img" in fileName
        ]

        if len(semantic_labels) == len(images):
          imgs.extend(images)
          labels.extend(semantic_labels)
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
    image_h, image_w, _ = tf.unstack(tf.shape(image))
    image_w = tf.cast(image_w, tf.float64)
    image_h = tf.cast(image_h, tf.float64)
    # Remove top 10 and bottom 10%
    cut_top = tf.cast(image_h * 0.1, tf.int32)
    crop_height = tf.cast(0.8 * image_h, tf.int32)

    # Calculate width of cropped image to have the right shape
    aspect_ratio = size[1] / size[0]
    crop_width = tf.cast(
        tf.cast(crop_height, tf.float32) * aspect_ratio, tf.int32)
    cut_left = tf.cast((image_w - tf.cast(crop_width, tf.float64)) / 2.,
                       tf.int32)

    cropped_image = tf.image.crop_to_bounding_box(image, cut_top, cut_left,
                                                  crop_height, crop_width)
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
