###########################################################################################
#    Contains all logic that is used to load images from the disk
###########################################################################################
import tensorflow as tf
import os
import random

class DataLoader:
    def __init__(self, workingDir, inputSize, outputSize = None, batch_size = 4, train_valid_split = 0.1):
        self.workingDir = workingDir
        self.batch_size = batch_size
        self.train_valid_split = train_valid_split


        self.inputSize = inputSize
        if outputSize is None:
            self.outputSize = inputSize
        else:
            self.outputSize = outputSize

    def parse_function(self, filename, label):
        image_string = tf.io.read_file(filename)

        image = tf.image.decode_png(image_string, channels=3)
        
        labels = tf.image.decode_png(tf.io.read_file(label), channels=1)
        
        #This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        cropped_image = tf.image.crop_to_bounding_box(image, 100,100,360,720)
        resized_image = tf.image.resize(cropped_image, self.inputSize)
        
        
        labels = tf.image.crop_to_bounding_box(labels, 100,100,360,720)
        labels = tf.image.resize(labels, self.inputSize, method="nearest")
        return resized_image, labels


    def train_preprocess(self, image, label):
        #image = tf.image.random_flip_left_right(image)

        #image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

        #Make sure the image is still in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label


    def getDataset(self):
        
        seed = random.randint(0,100)
        
        filenames = [self.workingDir + f+"/img.png" for f in os.listdir(self.workingDir)]
        random.seed(seed)
        random.shuffle(filenames)
        
        labels = [self.workingDir + f+"/semseg.png" for f in os.listdir(self.workingDir)]
        random.seed(seed)
        random.shuffle(labels)
        
        split_point = int(self.train_valid_split * len(filenames))
        
        
        filenames_train = filenames[0:split_point]
        filenames_valid = filenames[split_point+1:]
        
        labels_train = labels[0:split_point]
        labels_valid = labels[split_point+1:]
        
        return self.getDatasetForList(filenames_train, labels_train), self.getDatasetForList(filenames_valid, labels_valid)
        

    def getDatasetForList(self, imgs, labels):
        dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
        dataset = dataset.shuffle(len(imgs))
        dataset = dataset.map(self.parse_function, num_parallel_calls=4)
        dataset = dataset.map(self.train_preprocess, num_parallel_calls=4)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        
        return dataset