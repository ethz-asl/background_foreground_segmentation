import tensorflow as tf
from bfseg.utils.losses import ignorant_cross_entropy_loss


def test_ignorant_loss():
  # Generate random labels and predictions
  labels = tf.random.uniform(shape=[4, 120, 240, 1],
                             minval=0,
                             maxval=3,
                             dtype=tf.int64)
  prediction = tf.random.uniform(shape=[4, 120, 240, 2], dtype=tf.float64)

  # Loss for this prediction
  loss1 = ignorant_cross_entropy_loss(labels, prediction).numpy()

  # Get labels as one hot encoded tensor
  labels_as_tensor = tf.squeeze(tf.keras.backend.one_hot(labels, 3))

  # Pixels that should be ignored
  class_to_ignore = labels_as_tensor[..., 1]

  # Inverted pixels that should be ignored
  inverted_classes_to_ignore = tf.math.logical_not(
      tf.cast(class_to_ignore, tf.bool))

  # Perturb prediction such that all pixels that are assigned label 1 are now assigned label 0
  new_prediction_class_bg = tf.multiply(
      tf.cast(inverted_classes_to_ignore, tf.float64), prediction[..., 0])
  new_prediction_class_fg = prediction[..., 1]
  new_prediction_2 = tf.stack(
      [new_prediction_class_bg, new_prediction_class_fg], axis=-1)
  loss2 = ignorant_cross_entropy_loss(labels, new_prediction_2).numpy()

  # Make sure loss did not change
  assert loss1 == loss2

  # Perturb prediction such that all pixels that are assigned label 1 are now assigned label 2
  new_prediction_class_bg = prediction[..., 0]
  new_prediction_class_fg = tf.multiply(
      tf.cast(inverted_classes_to_ignore, tf.float64), prediction[..., 1]
  )  # tf.multiply(tf.cast(inverted_classes_to_ignore, tf.float64), prediction[...,1])
  new_prediction_3 = tf.stack(
      [new_prediction_class_bg, new_prediction_class_fg], axis=-1)
  loss3 = ignorant_cross_entropy_loss(labels, new_prediction_3).numpy()

  # Make sure loss did not change
  assert loss2 == loss3

  # Make sure loss changes if prediction are completely different and not only pixels for class 1 changes
  changed_prediction = prediction + tf.random.uniform(shape=[4, 120, 240, 2],
                                                      dtype=tf.float64)
  loss4 = ignorant_cross_entropy_loss(labels, changed_prediction).numpy()

  assert loss4 != loss1
