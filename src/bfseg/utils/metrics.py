import tensorflow as tf


class IgnorantDepthMAE(tf.keras.metrics.MeanAbsoluteError):
  """
    Ignorant wrapper for mean absolute error metric
  """

  def __init__(self):
    super().__init__()

  def update_state(self, depth_label, y_pred_depth, sample_weight=None):
    # Depth is predicted as inverse
    y_pred_depth = 10 / y_pred_depth
    depth_label = 10 / depth_label
    y_pred_depth_ignorant = tf.where(tf.math.is_nan(depth_label),
                                     tf.zeros_like(depth_label), y_pred_depth)
    depth_label = tf.where(tf.math.is_nan(depth_label),
                           tf.zeros_like(depth_label), depth_label)

    return super(IgnorantDepthMAE, self).update_state(depth_label, y_pred_depth)

  def result(self):
    return super(IgnorantDepthMAE, self).result()

  def reset_states(self):
    return super(IgnorantDepthMAE, self).reset_states()


class IgnorantMetricsWrapper(tf.keras.metrics.Metric):
  """
  Wraps any keras metric to ignore a specific class or balance the weights
  """

  def __init__(self, metric, balanced=False, class_to_ignore=1, num_classes=3):
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

  def __init__(self, class_to_ignore=1, num_classes=3):
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

  def __init__(self, class_to_ignore=1, num_classes=3):
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


def getIgnoreWeight(labels_one_hot, class_to_ignore):
  # invert class to be used as weights
  # This has a 1 for every pixel that has a valid ground truth label and a 0 for everything else
  ignore = labels_one_hot[..., class_to_ignore]
  return tf.cast(tf.math.logical_not(tf.cast(ignore, tf.bool)), tf.int32)
