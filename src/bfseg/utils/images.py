import tensorflow as tf


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
def augmentation(image, label, mask=None):
  # random flip
  if tf.random.uniform((1,)) < .5:
    image = tf.image.flip_left_right(image)
    label = tf.image.flip_left_right(label)
    if (mask is not None):
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
  if (mask is not None):
    return image, label, mask
  else:
    return image, label
