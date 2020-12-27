"""
    Contains all logic that is used to load images from the disk
"""
import tensorflow as tf
import os
import re
import random
import pandas


class DataLoader:

  def __init__(self,
               workingDir,
               inputSize,
               outputSize=None,
               validationDir=None,
               batchSize=4,
               shuffleBufferSize=64,
               validationMode="all",
               loadDepth=True,
               trainFilter = None,
               validationFilter = None,
               cropOptions={
                   'top': 0.1,
                   'bottom': 0.1
               }):
    """
      Dataloader to load datasets create with the datset creator
      Args:
          workingDir: Path to dataset images
          inputSize: Shape of the input image
          outputSize: Shape of the output image (labels)
          validationDir: Path to the validation images
          batchSize: Batch size to use
          shuffleBufferSize: Shuffle buffer batchsize
          validationMode: <all,CLA,ARCHE> which validation dataset to load
          loadDepth: whether to also load depth images
          trainFilter: Filter to filter training data by timestamp, camera etc.
                          e.g.
                          {
                                   'rgb_2': {
                                       'timestamp' : {
                                           'lower_bound' : 0,
                                           'upper_bound' : 1593603395.1234674
                                       }
                                   }
                               },
          validationFilter: see above. additionally has parameter max_count

          cropOptions: Dict specifing how much the image should be cropped. Numbers are treated as percentages
            e.g.: {top:0.1, bottom:0.5} crops the top 10% and bottom 50% of the image.
      """

    self.workingDir = workingDir
    self.batchSize = batchSize
    self.shuffleBufferSize = shuffleBufferSize
    self.inputSize = inputSize
    self.loadDepth = loadDepth
    self.cropOptions = cropOptions

    if outputSize is None:
      self.outputSize = inputSize
    else:
      self.outputSize = outputSize

    if validationDir is None:
      self.validationDir = workingDir
    else:
      self.validationDir = validationDir

    self.filenames, self.depths, self.labels = self.getImageDataFromPath(
        self.workingDir)


    if loadDepth:
      if len(self.depths) == 0:
        print("[WARNING] DID NOT FIND ANY DEPTH IMAGES!")
        self.depths = None
    else:
      self.depths = None



    self.validationFiles, self.validationDepth, self.validationLabels = self.getImageDataFromPath(self.validationDir)

    if validationFilter is not None:
        if validationMode is not "all":
          if validationMode is "ARCHE":
            # Ugly, change filenames in the hive data loader
            # Files with timestamp starting with _159 are from Arche
            self.validationFiles = [
                file for file in self.validationFiles if "_159" in file
            ]
            self.validationLabels = [
                file for file in self.validationLabels if "_159" in file
            ]

          elif validationMode is "CLA":
            # Ugly, change filenames in the hive data loader
            # Files with timestamp starting with _158 are from CLA
            self.validationFiles = [
                file for file in self.validationFiles if "_158" in file
            ]
            self.validationLabels = [
                file for file in self.validationLabels if "_158" in file
            ]
          else:
            print("Validation MODE", validationMode,
                  "is unknown. Going to use all validation data")

    print(f"Available Filenames. \n  Training:{len(self.filenames)} \n Validation:{len(self.validationFiles)}")

    if validationFilter is not None:
        self.validationFiles, self.validationLabels = self.applyFilterToValidFileNames(self.validationFiles,
                                                                                       self.validationLabels,
                                                                                       validationFilter)
    if trainFilter is not None:
        self.filenames, self.labels, self.depths = self.applyFilterToTrainFileNames(self.filenames, self.labels,
                                                                                    self.depths, trainFilter)
    self.size = len(self.filenames)
    self.validationSize = len(self.validationFiles)

    print(f"Loaded Filenames. \n  Training:{self.size} \n Validation:{self.validationSize}")

  def applyFilterToValidFileNames(self, filenames, labels, filter):
      """
      Filters the validation filenames using a timestamp/route filter supplied in the filter parameter
      Args:
          filenames: list with paths to validation images (validation filenames are from thehive and contain the timestamp as filename)
          labels: list with paths to labels
          filter: custom filter object. e.g.
          {
                'cam0' :  { 'timestamp' : { 'lower_bound': 0, 'upper_bound': 1593603395.150319 } },
                max_count: -1
          }

      Returns: filetered filenames and labels. Only files matched by the filter are returned.
      """
      filteredFilenames = []
      filteredLabels = []

      regex_pattern = "(.+)_(\d+\.\d+)_img\.png"

      for i, name in enumerate(filenames):
        match = re.search(regex_pattern, os.path.basename(name))
        route = match.group(1)
        timestamp = float(match.group(2))

        if route in filter.keys():
            if filter[route]['timestamp']['lower_bound'] < timestamp < filter[route]['timestamp']['upper_bound']:
                filteredFilenames.append(name)
                filteredLabels.append(labels[i])

      if "max_count" in filter.keys():
          # limit how many files should be in the validation set
          max_count = filter['max_count']
          # to not introduce bias, only take every nth images instead of randomly selecting max_count images
          keep_nth_image = len(filteredFilenames) // max_count
          return filteredFilenames[::keep_nth_image][0:max_count], filteredLabels[::keep_nth_image][0:max_count]

      return filteredFilenames, filteredLabels


  def applyFilterToTrainFileNames(self, filenames, labels, depths = None, filter = {}):
    """
       Filters the training filepaths using a timestamp/route filter supplied in the filter parameter
       Args:
           filenames: list with paths to training images
           labels: list with paths to labels
           depths: list with paths to depth images

           filter: custom filter object. e.g.
           {
                 'rgb_0' :  { 'timestamp' : { 'lower_bound': 0, 'upper_bound': 1593603395.150319 } },
                 'rgb_1' :  { 'timestamp' : { 'lower_bound': 0, 'upper_bound': 1593603395.150319 } },
           }

       Returns: filetered filenames, depths and labels. Only files matched by the filter are returned.

    """
    print("filter:", filter)
    filteredFilenames = []
    filteredLabels = []
    filteredDepths = [] if depths is not None else None

    # filenames have format: <route>_img_<number>_img
    regex_pattern = "_img_(\d+)_img\.png"

    # convert timestamps to image numbers.
    for filterRoute in filter.keys():
        # open info file that stores mapping timestap <-> image number
        info_file =  os.path.join(self.workingDir, f"{filterRoute}_info.txt")
        if not os.path.exists(info_file):
            info_file = os.path.join(self.workingDir, f"../{filterRoute}/{filterRoute}_info.txt")

        df = pandas.read_csv(info_file, header =None, sep =',|;', engine = "python")
        # only select image numbers that match timestamp
        valid_timestamps = (filter[filterRoute]['timestamp']['lower_bound'] <= df[1]) & (df[1] <= filter[filterRoute]['timestamp']['upper_bound'])
        valid_images = list(df[0][valid_timestamps])
        # cache request
        filter[filterRoute]['imageNumbers'] = valid_images

    for i, name in enumerate(filenames):
      basename = os.path.basename(name)
      # print("dealing with basename", basename)
      # extract image number from filename
      image_number = int(re.search(regex_pattern, basename).group(1))
      route = re.sub(regex_pattern, "", basename)

      if route in filter.keys():
        routeFilter = filter[route]
        validNumbers = routeFilter['imageNumbers']
        # print("checking image number", image_number)
        if image_number in validNumbers:
            # apply filter
            filteredFilenames.append(name)
            filteredLabels.append(labels[i])
            if depths is not None:
                filteredDepths.append(depths[i])
        else:
            print("image number", image_number, "was not in valid numbers", validNumbers)
      else:
        print("no filter found for ", route, " route will be ignored")

    return filteredFilenames, filteredLabels, filteredDepths


  def getImageDataFromPath(self, path):
    """ returns all input images and labels stored in the given 'path' folder.
                The folder structure looks like this:
                path/
                    img_0000/
                     - img_0000__img.[png/jpg]   <- Original image
                     - img_0000_semseg.png      <- Labels for Semantic Segmentation
                     - img_0000_distance.png    <- Labels for depth prediction
                     - ... additional information e.g. pc, pose
                    img_9999/
                      ....

                Currently the distance label is unused
                """
    # lists to return
    labels = []
    imgs = []
    depths = []

    # iterate through all imgXXX folders
    for image_folder in sorted(os.listdir(path)):
      image_folder_path = os.path.join(path, image_folder)
      # make sure it is folder
      if os.path.isdir(image_folder_path):
        # cache folder content (e.g. img.png, semseg.png)
        folder_content = sorted(os.listdir(image_folder_path))
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

        # count how many original images are there
        depth = [
            os.path.join(image_folder_path, fileName)
            for fileName in folder_content
            if image_folder + "_distance" in fileName
        ]

        if len(semantic_labels) == len(images):
          imgs.extend(images)
          labels.extend(semantic_labels)
          depths.extend(depth)
        else:
          print("WARNING! Label / Image missmatch in folder:",
                path + "image_folders")

    return imgs, depths, labels

  def parse_function(self, filename, label, *args):
    """ Read image and labels. Will crash if filetype is neither jpg nor png. """

    depthProvided = False
    depth = args[0] if len(args) == 1 else None
    print("label", label, "args", args, "filname", filename)

    if tf.io.is_jpeg(filename):
      image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
    else:
      image = tf.image.decode_png(tf.io.read_file(filename), channels=3)

    labels = tf.image.decode_png(tf.io.read_file(label), channels=1)

    if depth is not None:
      depths = tf.image.decode_png(tf.io.read_file(depth), channels=1)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    cropped_image = self.cropImageToInputSize(image, self.inputSize)
    cropped_semseg_labels = self.cropImageToInputSize(labels,
                                                      self.outputSize,
                                                      method="nearest")

    if depth is None:
      return cropped_image, cropped_semseg_labels

    depth_cropped = tf.cast(self.cropImageToInputSize(depths,
                                                      self.outputSize,
                                                      method="nearest"),
                            dtype=tf.float32)
    # Convert depth [0,255] to real distance [0m, 10m]
    depth_norm = ((tf.cast(depth_cropped, dtype=tf.float32) - 1.0) / 25.4)
    depth_norm_2 = tf.where(
        tf.equal(depth_cropped, tf.constant(0, dtype=tf.float32)),
        tf.constant(float('nan'), dtype=tf.float32), depth_norm)
    return cropped_image, {
        'depth': depth_norm_2,
        'semseg': cropped_semseg_labels,
        'combined': cropped_semseg_labels
    }

  def cropImageToInputSize(self, image, size, method="bilinear"):
    """ Crop image. Removes a little bit from the top of the image, as any won't have labels for this area """
    image_h, image_w, _ = tf.unstack(tf.shape(image))
    image_w = tf.cast(image_w, tf.float64)
    image_h = tf.cast(image_h, tf.float64)

    top_percentage = self.cropOptions['top']
    bot_percentage = self.cropOptions['bottom']

    # Remove top 10 and bottom 10%
    cut_top = tf.cast(image_h * top_percentage, tf.int32)
    crop_height = tf.cast((1 - top_percentage - bot_percentage) * image_h,
                          tf.int32)

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

  def train_preprocess(self, image, label, *args):
    """
                 Args:
                     image: keras tensor containing rgb image
                     label: keras tensor containing label image (1 channel)

                 Returns: randomly augmented image

                 """
    depth = args if len(args) == 1 else None

    # random data augmentation can be uncommented here
    # # Flip left right
    do_flip = tf.random.uniform([]) > 0.5
    image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image),
                   lambda: image)
    label = tf.cond(do_flip, lambda: tf.image.flip_left_right(label),
                   lambda: label)

    if depth is not None:
        depth = tf.cond(do_flip, lambda: tf.image.flip_left_right(depth),
                        lambda: depth)

    # image = tf.image.random_flip_left_right(image)
    # # Change brightness
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # # Change saturation
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    if depth is None:
        return image, label
    return image, label, depth

  def reduce_validation_labels(self, image, label):
    """
            Ground truth labels from the hive contain multiple classes.
            Reduce these multiple classes to background = 0, unknown = 1, foreground = 2
            Args:
                image: tensor size (None, img_h,img_w, 3)
                label: tensor size (None, img_h,img_w, 1)
            Returns: image, processed labels
            """

    label = tf.math.multiply(tf.cast(tf.cast(label, tf.bool), tf.int32), 2)
    return image, label

  def getDataset(self):
    """                Returns two tf.data.Datasets.
                First one contains images width self supervised labels generated from the pointcloud
                Second one contains images width ground truth annotations
                """

    return self.getTrainingDataset(), self.getValidationDataset()

  def getExportableDataset(self):
    """ Returns the dataset without shuffles, batches augmentation etc. """
    return tf.data.Dataset.from_tensor_slices((self.filenames, self.labels, self.depths))\
               .map(self.parse_function, num_parallel_calls=4) ,\
           tf.data.Dataset.from_tensor_slices((self.validationFiles, self.validationLabels))\
               .map(self.parse_function, num_parallel_calls=4)

  def getValidationDataset(self):
    """ Returns a tensorflow dataset based on list of filenames """
    # Passing validation labels as depth since they will get removed anyway
    return tf.data.Dataset.from_tensor_slices((self.validationFiles, self.validationLabels)) \
        .shuffle(self.validationSize) \
        .map(self.parse_function, num_parallel_calls=4) \
        .map(self.reduce_validation_labels, num_parallel_calls=4) \
        .batch(self.batchSize) \
        .prefetch(tf.data.experimental.AUTOTUNE)

  def getTrainingDataset(self):
    """ Returns a tensorflow dataset based on list of filenames """
    return tf.data.Dataset.from_tensor_slices((self.filenames, self.labels, self.depths)) \
        .shuffle(self.size) \
        .map(self.parse_function, num_parallel_calls=4) \
        .map(self.train_preprocess, num_parallel_calls=4) \
        .batch(self.batchSize) \
        .prefetch(tf.data.experimental.AUTOTUNE)
