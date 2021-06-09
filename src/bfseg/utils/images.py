import tensorflow as tf
import tensorflow_addons as tfa


@tf.function
def resize_with_crop(image, shape, method='bilinear'):
  """
  Resizes an image while maintaining aspect ratio by cropping away parts of the image.
  """
  target_h, target_w = shape
  target_aspect = tf.cast(target_w, tf.float32) / tf.cast(target_h, tf.float32)
  image_shape = tf.shape(image)
  image_h = tf.cast(image_shape[0], tf.float32)
  image_w = tf.cast(image_shape[1], tf.float32)
  input_aspect = image_w / image_h

  if input_aspect >= target_aspect:
    # image is too wide
    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=0,
        offset_width=tf.cast(.5 * (image_w - target_aspect * image_h) - .5,
                             tf.int32),
        target_height=image_shape[0],
        target_width=tf.cast(target_aspect * image_h, tf.int32))
  else:
    # image is too high
    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=tf.cast(.5 * (image_h - image_w / target_aspect) - .5,
                              tf.int32),
        offset_width=0,
        target_height=tf.cast(image_w / target_aspect, tf.int32),
        target_width=image_shape[1])

  return tf.image.resize(image, (target_h, target_w), method=method)


@tf.function
def augmentation(image, label):
  # make sure image is in float space
  image = tf.image.convert_image_dtype(image, tf.float32)
  # random flip
  if tf.random.uniform((1,)) < .5:
    image = tf.image.flip_left_right(image)
    label = tf.image.flip_left_right(label)
  # brightness
  image = tf.image.random_brightness(image, max_delta=0.2)
  # hue
  image = tf.image.random_hue(image, max_delta=.1)
  return image, label


@tf.function
def augmentation_with_mask(image, label, mask):
  # make sure image is in float space
  image = tf.image.convert_image_dtype(image, tf.float32)
  # random flip
  if tf.random.uniform((1,)) < .5:
    image = tf.image.flip_left_right(image)
    label = tf.image.flip_left_right(label)
    # Note: it is crucial that `expand_dims` is used, otherwise, since `mask`
    # is in (B, H, W) format, rather than (B, H, W, 1). `flip_left_right`
    # would wrongly flip the image upside down.
    mask = tf.image.flip_left_right(tf.expand_dims(mask, axis=-1))
    # Bring the mask back to its initial shape.
    mask = tf.squeeze(mask, axis=-1)
  # brightness
  image = tf.image.random_brightness(image, max_delta=0.2)
  # hue
  image = tf.image.random_hue(image, max_delta=.1)
  return image, label, mask

@tf.function
def augmentation_with_mask_depth(image, labels, masks):
  # make sure image is in float space
  image = tf.image.convert_image_dtype(image, tf.float32)

  # random flip
  if tf.random.uniform((1,)) < .5:
    # unpack
    mask_seg = masks['seg_mask']
    mask_depth = masks['depth_mask']
    label_seg = labels['seg_label']
    label_depth = labels['depth_label']

    image = tf.image.flip_left_right(image)
    label_seg = tf.image.flip_left_right(label_seg)
    label_depth = tf.image.flip_left_right(label_depth)
    # Note: it is crucial that `expand_dims` is used, otherwise, since `mask`
    # is in (B, H, W) format, rather than (B, H, W, 1). `flip_left_right`
    # would wrongly flip the image upside down.
    mask_seg = tf.image.flip_left_right(tf.expand_dims(mask_seg, axis=-1))
    mask_depth = tf.image.flip_left_right(tf.expand_dims(mask_depth, axis=-1))
    # Bring the mask back to its initial shape.
    mask_seg = tf.squeeze(mask_seg, axis=-1)
    mask_depth = tf.squeeze(mask_depth, axis=-1)
    # pack 
    masks = {'seg_mask': mask_seg, 'depth_mask': mask_depth}
    labels = {'seg_label': label_seg, 'depth_label': label_depth}

  # brightness
  image = tf.image.random_brightness(image, max_delta=0.2)
  # hue
  image = tf.image.random_hue(image, max_delta=.1)
  return image, labels, masks

@tf.function
def preprocess_median_full_with_mask_depth(image, labels, masks):
  print("-------Inside preprocessing_median_full_with_mask_depth")
  label_seg = labels['seg_label']
  label_depth = labels['depth_label']
  mask_seg = masks['seg_mask']
  mask_depth = masks['depth_mask']
  print("Mast_seg shape 1: {}".format(mask_seg.shape))
  print("Label_seg shape 1: {}".format(label_seg.shape))
  # Preprocess Labels with median filter and remove unknown category
  label_seg_median = tfa.image.median_filter2d(label_seg, 11)
  label_seg_median_full = tf.where(
        tf.equal(label_seg_median, tf.constant(2, dtype=tf.uint8)),
        tf.constant(0, dtype=tf.uint8), label_seg_median)
  mask_seg = (tf.not_equal(label_seg, -1)) # All true.
  mask_seg = tf.squeeze(mask_seg, axis=-1)
  print("Label_seg_median: {}".format(label_seg_median.shape))
  print("Label_seg_median_full: {}".format(label_seg_median_full.shape))
  print("Mast_seg shape 2: {}".format(mask_seg.shape))

  labels = {'seg_label': label_seg_median_full, 'depth_label': label_depth}
  masks = {'seg_mask': mask_seg, 'depth_mask': mask_depth}
  return image, labels, masks

@tf.function
def preprocess_median_full_with_mask(image, label, mask):
  print("-------Inside preprocessing_median_full_with_mask")

  label_seg_median = tfa.image.median_filter2d(label, 11)
  label_seg_median_full = tf.where(
        tf.equal(label_seg_median, tf.constant(2, dtype=tf.uint8)),
        tf.constant(0, dtype=tf.uint8), label_seg_median)
  mask = tf.not_equal(label, -1)  # All true.
  mask = tf.squeeze(mask, axis=-1)

  return image, label, mask
