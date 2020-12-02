import tensorflow as tf


class IgnorantBalancedAccuracyMetric(tf.keras.metrics.Metric):
  """
  Accuracy function that ignores a class with a given label and balances the weights.
  e.g. if the GT has 90% background and 10% foreground, a pixel that is correctly predicted as background counts less
  than a pixel that is correctly predicted as foreground.

  Randomly Predicting 90%foreground and 10% background will produce a balanced accuracy of 50%
  """

  def __init__(self, class_to_ignore=1, class_cnt=3):
    super().__init__()
    self.k_acc = tf.keras.metrics.Accuracy()
    self.class_to_ignore = class_to_ignore
    self.class_cnt = class_cnt

  def update_state(self, y_true, y_pred, sample_weight=None):
    return _accuracy(y_true, y_pred, True, self.class_to_ignore, self.class_cnt,
                     self.k_acc)

  def result(self):
    return self.k_acc.result()

  def reset_states(self):
    return self.k_acc.reset_states()


class IgnorantAccuracyMetric(tf.keras.metrics.Metric):
  """
  Accuracy function that ignores a class with a given label
  """

  def __init__(self, class_to_ignore=1, class_cnt=3):
    super().__init__()
    self.k_acc = tf.keras.metrics.Accuracy()
    self.class_to_ignore = class_to_ignore
    self.class_cnt = class_cnt

  def update_state(self, y_true, y_pred, sample_weight=None):
    return _accuracy(y_true, y_pred, False, self.class_to_ignore,
                     self.class_cnt, self.k_acc)

  def result(self):
    return self.k_acc.result()

  def reset_states(self):
    return self.k_acc.reset_states()


def _accuracy(y_true, y_pred, balanced, class_to_ignore, num_classes,
              keras_accuracy):

  # convert true labels to one hot encoded images
  labels_one_hot = tf.squeeze(tf.keras.backend.one_hot(y_true, 3))

  # Remove class that should be ignored from one hot encoding
  y_true_one_hot_no_ignore = tf.stack([
      labels_one_hot[..., _class]
      for _class in range(num_classes)
      if _class != class_to_ignore
  ],
                                      axis=-1)

  # Transform one hot encoding back to categorical
  y_true_back = tf.cast(tf.math.argmax(y_true_one_hot_no_ignore, axis=-1),
                        tf.int64)

  if balanced:
    weights = getBalancedWeight(y_true, labels_one_hot, class_to_ignore,
                                num_classes)
  else:
    weights = getIgnoreWeight(
        labels_one_hot,
        class_to_ignore,
    )

  return keras_accuracy.update_state(tf.squeeze(y_true_back),
                                     tf.squeeze(tf.argmax(y_pred, axis=-1)),
                                     sample_weight=tf.squeeze(weights))


def getBalancedWeight(labels, labels_one_hot, class_to_ignore, num_classes):
  weight_tensor = tf.cast(tf.zeros_like(labels), tf.float32)

  for i in range(num_classes):
    if i == class_to_ignore:
      continue
    # Calculate how many pixel of this class occur
    frequency = tf.math.scalar_mul(
        1 / tf.reduce_sum(tf.cast(labels == i, tf.float32)), labels_one_hot[...,
                                                                            i])
    frequency = tf.expand_dims(frequency, -1)
    # add to weight tensor
    weight_tensor = tf.math.add(weight_tensor, frequency)
  return weight_tensor


def getIgnoreWeight(labels_one_hot, class_to_ignore):
  # invert class to be used as weights
  # This has a 1 for every pixel that has a valid ground truth label and a 0 for everything else
  ignore = labels_one_hot[..., class_to_ignore]
  return tf.cast(tf.math.logical_not(tf.cast(ignore, tf.bool)), tf.int32)
