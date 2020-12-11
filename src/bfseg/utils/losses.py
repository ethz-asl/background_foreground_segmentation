import tensorflow as tf

from bfseg.utils.metrics import  getBalancedWeight, getIgnoreWeight
def ignorant_cross_entropy_loss(y_true,
                                y_pred,
                                class_to_ignore=1,
                                num_of_classes=3):
  """
    Loss function that ignores all classes with label class_to_ignore.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_to_ignore: Class number from ground truth which should be ignored
        num_classes: how many classes there are

    Returns: Cross entropy loss where ground truth labels that have class 'class_to_ignore' are ignored
    """

  """
    Loss function that ignores all classes with label class_to_ignore.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_to_ignore: Class number from ground truth which should be ignored
        num_classes: how many classes there are

    Returns: Cross entropy loss where ground truth labels that have class 'class_to_ignore' are ignored
    """
  # convert true labels to one hot encoded images
  labels_one_hot = tf.keras.backend.one_hot(y_true, 3)
  # Remove class that should be ignored from one hot encoding
  y_true_one_hot_no_ignore = tf.stack([
      labels_one_hot[..., _class]
      for _class in range(num_of_classes)
      if _class != class_to_ignore
  ],
      axis=-1)

  # Transform one hot encoding back to categorical
  y_true_back = tf.cast(tf.math.argmax(y_true_one_hot_no_ignore, axis=-1),
                        tf.int64)

  weights = getIgnoreWeight(
          labels_one_hot,
          class_to_ignore,

      )

  scce = tf.keras.losses.SparseCategoricalCrossentropy()
  return scce(y_true_back, y_pred, sample_weight=weights)

def ignorant_balanced_cross_entropy_loss(y_true,
                                y_pred,
                                class_to_ignore=1,
                                num_of_classes=3):
  """
    Loss function that ignores all classes with label class_to_ignore.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_to_ignore: Class number from ground truth which should be ignored
        num_classes: how many classes there are

    Returns: Cross entropy loss where ground truth labels that have class 'class_to_ignore' are ignored
    """
  # convert true labels to one hot encoded images
  labels_one_hot = tf.keras.backend.one_hot(y_true, 3)
  # Remove class that should be ignored from one hot encoding
  y_true_one_hot_no_ignore = tf.stack([
      labels_one_hot[..., _class]
      for _class in range(num_of_classes)
      if _class != class_to_ignore
  ],
      axis=-1)

  # Transform one hot encoding back to categorical
  y_true_back = tf.cast(tf.math.argmax(y_true_one_hot_no_ignore, axis=-1),
                        tf.int64)


  weights = getBalancedWeight(y_true, labels_one_hot, class_to_ignore,
                                  num_of_classes, normalize = False)

  scce = tf.keras.losses.SparseCategoricalCrossentropy()
  return scce(y_true_back, y_pred, sample_weight=weights)

def combined_loss(pseudo_label_weight, threshold, balanced):
    def loss(y_true, y_pred):
        p_labels = pseudo_labels(y_true, y_pred, threshold)

        if balanced:
            p_labels_loss = ignorant_cross_entropy_loss(p_labels, y_pred)
            meshdist_loss = ignorant_cross_entropy_loss(y_true,y_pred)
        else:
            p_labels_loss = ignorant_balanced_cross_entropy_loss(p_labels, y_pred)
            meshdist_loss = ignorant_balanced_cross_entropy_loss(y_true, y_pred)

        return pseudo_label_weight* p_labels_loss + (1-pseudo_label_weight)*meshdist_loss
    return loss

def pseudo_labels(y_true, y_pred, threshold):
    """ Converts a given prediction into pseudo labels, classes that have a prediction smaller than the provided threshold
    will be assigned the 1 (ignore) label. """
    # Boolean mask. one where prediction is above threshold, false where it is not
    believe = tf.greater_equal(tf.reduce_max(y_pred, axis=-1), threshold)
    # Convert prediction into labels. [0,2]
    assignment = tf.scalar_mul(2,tf.argmax(y_pred, axis=-1))
    # Assign 1 to all predictions where the prediction was not certain enough
    pseudo_labels = tf.where(believe, assignment, tf.ones_like(assignment))
    return pseudo_labels

