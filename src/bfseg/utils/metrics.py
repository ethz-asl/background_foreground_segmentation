import tensorflow as tf


class IgnorantMetricsWrapper(tf.keras.metrics.Metric):
  """
  Wraps any keras metric to ignore a specific class or balance the weights
  """

  def __init__(self,
               metric,
               balanced=False,
               class_to_ignore=1,
               num_classes=3,
               **kwargs):
    super().__init__()
    self.metric = metric
    self.class_to_ignore = class_to_ignore
    self.num_of_classes = num_classes
    self.balanced = balanced

  def update_state(self, y_true, y_pred, sample_weight=None):

    # convert true labels to one hot encoded images
    labels_one_hot = tf.keras.backend.one_hot(y_true, self.num_of_classes)
    # Remove "unknown" class from groundtruth
    classes_to_keep = tf.constant([
        idx_class for idx_class in range(self.num_of_classes)
        if idx_class != self.class_to_ignore
    ])
    y_true_one_hot_no_ignore = tf.gather(labels_one_hot,
                                         classes_to_keep,
                                         axis=-1)
    # Transform one hot encoding back to categorical
    y_true_no_unknown = tf.math.argmax(y_true_one_hot_no_ignore, axis=-1)

    if self.balanced:
      weights = getBalancedWeight(y_true, labels_one_hot, self.class_to_ignore,
                                  self.num_of_classes)
    else:
      weights = getIgnoreWeight(
          labels_one_hot,
          self.class_to_ignore,
      )

    return self.metric.update_state(y_true_no_unknown,
                                    tf.expand_dims(tf.argmax(y_pred, axis=-1),
                                                   -1),
                                    sample_weight=weights)

  def result(self):
    return self.metric.result()

  def reset_states(self):
    return self.metric.reset_states()


class IgnorantBalancedMeanIoU(IgnorantMetricsWrapper):

  def __init__(self, class_to_ignore=1, num_classes=3):
    super().__init__(tf.keras.metrics.MeanIoU(num_classes=2),
                     balanced=True,
                     num_classes=num_classes,
                     class_to_ignore=class_to_ignore)


class IgnorantMeanIoU(IgnorantMetricsWrapper):

  def __init__(self, class_to_ignore=1, num_classes=3, **kwargs):
    super().__init__(tf.keras.metrics.MeanIoU(num_classes=2),
                     balanced=False,
                     num_classes=num_classes,
                     class_to_ignore=class_to_ignore)


class IgnorantBalancedAccuracyMetric(IgnorantMetricsWrapper):
  """
  Accuracy function that ignores a class with a given label and balances the weights.
  e.g. if the GT has 90% background and 10% foreground, a pixel that is correctly
  predicted as background counts less than a pixel that is correctly predicted as
  foreground.

  Randomly Predicting 90%foreground and 10% background will produce a balanced accuracy
  of 50%
  """

  def __init__(self, class_to_ignore=1, num_classes=3):
    super().__init__(tf.keras.metrics.Accuracy(),
                     balanced=True,
                     num_classes=num_classes,
                     class_to_ignore=class_to_ignore)


class IgnorantAccuracyMetric(IgnorantMetricsWrapper):
  """
  Accuracy function that ignores a class with a given label
  """

  def __init__(self, class_to_ignore=1, num_classes=3, **kwargs):
    super().__init__(tf.keras.metrics.Accuracy(),
                     num_classes=num_classes,
                     class_to_ignore=class_to_ignore)


def getBalancedWeight(labels,
                      labels_one_hot,
                      class_to_ignore,
                      num_classes,
                      normalize=True):
  weight_tensor = tf.cast(tf.zeros_like(labels), tf.float32)
  for i in range(num_classes):
    if i == class_to_ignore:
      continue
    # Calculate how many pixel of this class occur
    frequency = tf.math.scalar_mul(
        1 / tf.reduce_sum(tf.cast(labels == i, tf.float32)), labels_one_hot[...,
                                                                            i])
    # add to weight tensor
    if not normalize:
      frequency *= tf.reduce_sum(tf.cast(labels, tf.float32))
    weight_tensor = tf.math.add(weight_tensor, frequency)

  # tf.print("freq:", tf.unique(tf.reshape(weight_tensor, [-1])))
  # remove nan values if there are any
  weight_tensor = tf.where(tf.math.is_nan(weight_tensor),
                           tf.zeros_like(weight_tensor), weight_tensor)

  return weight_tensor


def get_balanced_weights(labels, num_classes):
  r"""Reproduces the behavior of `getBalancedWeight` for labels that have
  already been pre-filtered to remove the unknown class.

  Args:
    labels (tf.Tensor): Label tensor.
    num_classes (int): Number of classes.

  Returns:
    weight_tensor (tf.Tensor): Weight tensor. Same shape as label, but with each
      pixel `i` having weight `1 / num_pixels_same_class`, where
      `num_pixels_same_class` is the total number of pixels that have the same
      class as `i` over the input batch.
  """
  weight_tensor = tf.cast(tf.zeros_like(labels), tf.float32)

  for curr_class in range(num_classes):
    # Calculate how many pixels of this class occur.
    mask_curr_class = tf.cast(labels == curr_class, tf.float32)
    inverse_frequency = 1 / tf.reduce_sum(mask_curr_class)
    frequency = tf.math.scalar_mul(inverse_frequency, mask_curr_class)
    # Add to weight tensor.
    weight_tensor = tf.math.add(weight_tensor, frequency)

  # Remove nan values if there are any.
  weight_tensor = tf.where(tf.math.is_nan(weight_tensor),
                           tf.zeros_like(weight_tensor), weight_tensor)

  return weight_tensor


def getIgnoreWeight(labels_one_hot, class_to_ignore):
  # invert class to be used as weights
  # This has a 1 for every pixel that has a valid ground truth label and a 0 for everything else
  ignore = labels_one_hot[..., class_to_ignore]
  return tf.cast(tf.math.logical_not(tf.cast(ignore, tf.bool)), tf.int32)
