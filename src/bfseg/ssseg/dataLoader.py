###########################################################################################
#    Contains all logic that is used to load images from the disk
###########################################################################################
import tensorflow as tf
import os
import random


class DataLoader:
    def __init__(self, workingDir, inputSize, outputSize=None, validationDir=None, batchSize=4, shuffleBufferSize=64):
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
        self.validationFiles, self.validationLabels = self.getImageDataFromPath(self.validationDir)

        self.size = len(self.filenames)

    def getImageDataFromPath(self, path):
        labels = []
        imgs = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if ".txt" in file:
                    break
                if "label" in file or "semseg" in file:
                    labels.append(os.path.join(root, file))
                elif "img" in file:
                    imgs.append(os.path.join(root, file))

        return imgs, labels

    def parse_function(self, filename, label):
        # Read image and labels. Will crash if filetype is neither jpg nor png.
        if tf.io.is_jpeg(filename):
            image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
        else:
            image = tf.image.decode_png(tf.io.read_file(filename), channels=3)

        labels = tf.image.decode_png(tf.io.read_file(label), channels=1)

        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Make sure to NOT mess up the labels thus use nearest neighbour interpolation
        return self.cropImageToInputSize(image, self.inputSize), self.cropImageToInputSize(labels, self.outputSize,
                                                                                           method="nearest")

    def cropImageToInputSize(self, image, size, method="bilinear"):
        # Crop image. Removes a little bit from the top of the image, as any won't have labels for this area
        # TODO do not hardcode, make image size dependant
        image = tf.image.resize(image, [600, 900], method=method)
        cropped_image = tf.image.crop_to_bounding_box(image, 100, 0, 480, 640)
        # Resize it to desired input size of the network
        return tf.image.resize(cropped_image, size, method=method)

    def train_preprocess(self, image, label):
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
        # Todo return validation data set from the hive not None
        return self.getDatasetForList(self.filenames, self.labels), self.getDatasetForList(self.validationFiles, self.validationLabels)

        # Old code to generate random training and validation split
        # seed = random.randint(0, 100)
        #
        # filenames = [self.workingDir + f + "/img.png" for f in os.listdir(self.workingDir) if ".txt" not in f]
        # random.seed(seed)
        # random.shuffle(filenames)
        #
        # labels = [self.workingDir + f + "/semseg.png" for f in os.listdir(self.workingDir) if ".txt" not in f]
        # random.seed(seed)
        # random.shuffle(labels)
        #
        # split_point = int(self.train_valid_split * len(filenames))
        #
        # filenames_train = filenames[0:split_point]
        # filenames_valid = filenames[split_point + 1:]
        #
        # labels_train = labels[0:split_point]
        # labels_valid = labels[split_point + 1:]
        #
        # return self.getDatasetForList(filenames_train, labels_train), self.getDatasetForList(filenames_valid,
        #                                                                                      labels_valid)

    def getDatasetForList(self, imgs, labels):
        # Returns a tensorflow dataset based on list of filenames
        return tf.data.Dataset.from_tensor_slices((imgs, labels)) \
            .shuffle(len(imgs)) \
            .map(self.parse_function, num_parallel_calls=4) \
            .map(self.train_preprocess, num_parallel_calls=4) \
            .cache() \
            .shuffle(self.shuffleBufferSize) \
            .batch(4) \
            .prefetch(tf.data.experimental.AUTOTUNE)

