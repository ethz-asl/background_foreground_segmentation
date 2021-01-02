import tensorflow as tf

from bfseg.utils.metrics import getBalancedWeight, getIgnoreWeight


def ignorant_depth_loss(depth_label, y_pred_depth):
  """
  wrapper to mask all "NaN" values in depth
  """
  y_pred_depth_ignorant = tf.where(tf.math.is_nan(depth_label),
                                   tf.zeros_like(depth_label), y_pred_depth)
  depth_label = tf.where(tf.math.is_nan(depth_label),
                         tf.zeros_like(depth_label), depth_label)

  return depth_loss_function(
      depth_label, y_pred_depth_ignorant
  )  # tf.keras.losses.ber(depth_label, y_pred_depth_ignorant)


def depth_loss_function(y_true, y_pred, theta=10, maxDepthVal=1000.0 / 10.0):
  """ Loss Function from DenseDepth paper.
    Code taken from here https://github.com/ialhashim/DenseDepth/blob/master/loss.py
  """

  # Point-wise depth
  l_depth = tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true),
                                  axis=-1)
  # Edges
  dy_true, dx_true = tf.image.image_gradients(y_true)
  dy_pred, dx_pred = tf.image.image_gradients(y_pred)
  l_edges = tf.keras.backend.mean(tf.keras.backend.abs(dy_pred - dy_true) +
                                  tf.keras.backend.abs(dx_pred - dx_true),
                                  axis=-1)

  # Structural similarity (SSIM) index
  l_ssim = tf.keras.backend.clip(
      (1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

  # Weights
  w1 = 1.0
  w2 = 1.0
  w3 = theta

  return (w1 * l_ssim) + (w2 * tf.keras.backend.mean(l_edges)) + (
      w3 * tf.keras.backend.mean(l_depth))


def smooth_consistency_loss(depth_pred, y_pred_semantic, class_number=0):
  """
    Makes sure the semantic and depth prediction match. e.q. there are not too many edges inside a
    segmentation mask.

    Taken from this paper:
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Towards_Scene_Understanding_Unsupervised_Monocular_Depth_Estimation_With_Semantic-Aware_Representation_CVPR_2019_paper.pdf
  """

  phi = tf.cast(tf.math.equal(y_pred_semantic, class_number), dtype=tf.float32)
  phi_x = tf.roll(phi, 1, axis=-3)
  phi_y = tf.roll(phi, 1, axis=-2)
  depth_x = tf.roll(depth_pred, 1, axis=-3)
  depth_y = tf.roll(depth_pred, 1, axis=-2)
  diffx = tf.multiply(tf.math.abs(depth_pred - depth_x), 1 - tf.math.abs(
      (phi - phi_x)))
  diffx_no_nan = tf.where(tf.math.is_nan(diffx), tf.zeros_like(diffx), diffx)

  diffy = tf.multiply(tf.math.abs(depth_pred - depth_y), 1 - tf.math.abs(
      (phi - phi_y)))
  diffy_no_nan = tf.where(tf.math.is_nan(diffy), tf.zeros_like(diffy), diffy)

  return tf.keras.backend.mean(diffx_no_nan + diffy_no_nan)


def reduceGroundTruth(y_true, class_to_ignore=1, num_of_classes=3):
  """ convert true labels to one hot encoded images """
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
  return y_true_back, weights


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

  weights = getBalancedWeight(y_true,
                              labels_one_hot,
                              class_to_ignore,
                              num_of_classes,
                              normalize=False)

  scce = tf.keras.losses.SparseCategoricalCrossentropy()
  return scce(y_true_back, y_pred, sample_weight=weights)


def comdined_pseudo_label_loss(pseudo_label_weight, threshold, balanced):
  """ returns a*L_pseudoLabel() + (1-a)*L_cross_entropy """

  def loss(y_true, y_pred):
    # generate pseudo labels
    p_labels = pseudo_labels(y_true, y_pred, threshold)

    p_labels_loss = ignorant_cross_entropy_loss(
        p_labels, y_pred) if balanced \
      else ignorant_balanced_cross_entropy_loss( p_labels, y_pred)

    meshdist_loss = ignorant_cross_entropy_loss(y_true, y_pred) if balanced \
      else ignorant_balanced_cross_entropy_loss(  y_true, y_pred)

    return pseudo_label_weight * p_labels_loss + (
        1 - pseudo_label_weight) * meshdist_loss

  return loss


def pseudo_labels(y_true, y_pred, threshold):
  """ Converts a given prediction into pseudo labels, classes that have a confidence smaller than the provided threshold
    will be assigned the 1 (ignore) label. """
  # Boolean mask. one where prediction is above threshold, false where it is not
  believe = tf.greater_equal(tf.reduce_max(y_pred, axis=-1), threshold)
  # Convert prediction into labels. [0,2]
  assignment = tf.scalar_mul(2, tf.argmax(y_pred, axis=-1))
  # Assign 1 to all predictions where the prediction was not certain enough
  pseudo_labels = tf.where(believe, assignment, tf.ones_like(assignment))
  return pseudo_labels
