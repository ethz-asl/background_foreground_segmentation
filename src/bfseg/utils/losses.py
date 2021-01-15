import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


def ignorant_cross_entropy_loss(y_true,
                                y_pred,
                                class_to_ignore=1,
                                num_classes=3,
                                from_logits=False):
  """
    Loss function that ignores all classes with label class_to_ignore.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_to_ignore: Class number from ground truth which should be ignored
        num_classes: how many classes there are
        from_logits: set to True if y_pred are logits instead of softmax output. This
          gives better numerical stability.

    Returns: Cross entropy loss where ground truth labels that have class 'class_to_ignore' are ignored
    """

  # convert true labels to one hot encoded images
  data = tf.squeeze(tf.keras.backend.one_hot(y_true, 3))
  # extracts classes that should be ignored
  ignore = data[..., class_to_ignore]
  # invert class to be used as weights
  # This has a 1 for every pixel that has a valid ground truth label and a 0 for everything else
  ignore_inverted = tf.cast(tf.math.logical_not(tf.cast(ignore, tf.bool)),
                            tf.int32)

  # Remove class that should be ignored from one hot encoding
  y_true_one_hot_no_ignore = tf.stack([
      data[..., _class]
      for _class in range(num_classes)
      if _class != class_to_ignore
  ],
                                      axis=-1)

  # Transform one hot encoding back to categorical
  y_true_back = tf.cast(tf.math.argmax(y_true_one_hot_no_ignore, axis=-1),
                        tf.int64)

  scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
  return scce(y_true_back, y_pred, sample_weight=ignore_inverted)


class IgnorantCrossEntropyLoss(LossFunctionWrapper):
  """
  Wraps ignorant_cross_entropy_loss into an object to pass arguments at construction.
  """

  def __init__(self, class_to_ignore=1, num_classes=3, from_logits=False):
    super().__init__(ignorant_cross_entropy_loss,
                     class_to_ignore=class_to_ignore,
                     num_classes=num_classes,
                     from_logits=from_logits,
                     name='ignorant_cross_entropy_loss')
