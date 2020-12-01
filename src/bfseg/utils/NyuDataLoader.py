import tensorflow as tf
import tensorflow_datasets as tfds
from bfseg.data.nyu.Nyu_depth_v2_labeled import NyuDepthV2Labeled
import segmentation_models as sm
from bfseg.utils.metrics import IgnorantAccuracyMetric
from tensorflow import keras
import tensorflow.keras.preprocessing.image as Image
import os


class NyuDataLoader():

  def __init__(self, batch_size, shape):
    self.batch_size = batch_size
    self.shape = shape

  @tf.function
  def normalize_img(self, image, label):
    """Normalizes images: `uint8` -> `float32`."""
    # image = tf.image.resize(image,(32,64))
    # label = tf.cast(label, tf.uint8)
    # combine_label = tf.cast(tf.math.logical_or(tf.math.equal(label,4),(tf.logical_or(tf.math.equal(label,11),tf.math.equal(label,21)))),tf.uint8)
    # combine_label= tf.expand_dims(combine_label,axis=2)
    label = tf.expand_dims(label, axis=2)
    # print(label.shape)
    image = tf.cast(image, tf.float32) / 255.
    # Make sure to NOT mess up the labels thus use nearest neighbour interpolation
    input_size = (self.shape[1], self.shape[0])
    return self.cropImageToInputSize(
        image, input_size), self.cropImageToInputSize(label,
                                                      input_size,
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

  def create_mask(self, pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

  def getDataSets(self,):
    train_ds, train_info = tfds.load(
        'NyuDepthV2Labeled',
        split='train[:80%]',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    val_ds, val_info = tfds.load(
        'NyuDepthV2Labeled',
        split='train[80%:]',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    test_ds, test_info = tfds.load(
        'NyuDepthV2Labeled',
        split='test',
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    train_ds = train_ds.map(normalize_img,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(
        int(train_info.splits['train'].num_examples * 0.8))
    # train_ds = train_ds.shuffle(int(train_info.splits['train'].num_examples*0.8))
    train_ds = train_ds.batch(self.batch_size).repeat()
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    val_ds = val_ds.map(self.normalize_img,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(self.batch_size)
    val_ds = val_ds.cache()

    test_ds = test_ds.map(self.normalize_img,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(self.batch_size)
    test_ds = test_ds.cache()

    return train_ds, train_info, val_ds, val_info, test_ds, test_info
