import tensorflow as tf
from bfseg.utils.utils import IgnorantAccuracyMetric, IgnorantBalancedAccuracyMetric


def test_numeric_accuracy():
  """ Test accuracy with given values """

  ignorant_accuracy = IgnorantAccuracyMetric()
  weighted_ignorant_accuracy = IgnorantBalancedAccuracyMetric()

  labels = tf.constant([0, 0, 0, 0, 1, 2, 2, 2, 2], shape=(1, 3, 3, 1))

  prediction_cat = tf.constant([0, 0, 0, 0, 0, 1, 1, 1, 1], shape=(1, 3, 3, 1))
  prediction_one_hot = tf.squeeze(tf.keras.backend.one_hot(prediction_cat, 2))
  ignorant_accuracy.update_state(labels, prediction_one_hot)
  acc_correct = ignorant_accuracy.result().numpy()
  ignorant_accuracy.reset_states()
  assert acc_correct == 1

  prediction_wrong = tf.squeeze(
      tf.keras.backend.one_hot(
          tf.constant([1, 1, 1, 1, 0, 0, 0, 0, 0], shape=(1, 3, 3, 1)), 2))
  ignorant_accuracy.update_state(labels, prediction_wrong)
  acc_wrong = ignorant_accuracy.result().numpy()
  ignorant_accuracy.reset_states()
  assert acc_wrong == 0

  prediction_not_correct = tf.squeeze(
      tf.keras.backend.one_hot(
          tf.constant([0, 0, 0, 1, 0, 1, 1, 0, 1], shape=(1, 3, 3, 1)), 2))
  ignorant_accuracy.update_state(labels, prediction_not_correct)
  acc_not_correct = ignorant_accuracy.result().numpy()
  ignorant_accuracy.reset_states()
  assert acc_not_correct == 6 / 8


def test_numeric_accuracy_weighted():
  """ Test weighted accuracy with given values """

  ignorant_accuracy = IgnorantAccuracyMetric()
  weighted_ignorant_accuracy = IgnorantBalancedAccuracyMetric()

  labels = tf.constant([0, 0, 0, 0, 1, 2, 2, 2, 2], shape=(1, 3, 3, 1))

  prediction_cat = tf.constant([0, 0, 0, 0, 0, 1, 1, 1, 1], shape=(1, 3, 3, 1))
  prediction_one_hot = tf.squeeze(tf.keras.backend.one_hot(prediction_cat, 2))
  weighted_ignorant_accuracy.update_state(labels, prediction_one_hot)
  acc_correct = weighted_ignorant_accuracy.result().numpy()
  weighted_ignorant_accuracy.reset_states()
  assert acc_correct == 1

  prediction_wrong = tf.squeeze(
      tf.keras.backend.one_hot(
          tf.constant([1, 1, 1, 1, 0, 0, 0, 0, 0], shape=(1, 3, 3, 1)), 2))
  weighted_ignorant_accuracy.update_state(labels, prediction_wrong)
  acc_wrong = weighted_ignorant_accuracy.result().numpy()
  weighted_ignorant_accuracy.reset_states()
  assert acc_wrong == 0

  prediction_not_correct = tf.squeeze(
      tf.keras.backend.one_hot(
          tf.constant([0, 0, 0, 1, 0, 1, 1, 0, 1], shape=(1, 3, 3, 1)), 2))
  acc_not_correct = weighted_ignorant_accuracy(labels,
                                               prediction_not_correct).numpy()
  assert acc_not_correct == 6 / 8


def test_numeric_accuracy_weighted_unbalanced():
  """ Test weighted accuracy with given values on unbalanced labels."""

  ignorant_accuracy = IgnorantAccuracyMetric()
  weighted_ignorant_accuracy = IgnorantBalancedAccuracyMetric()

  # unbalanced labels ratio 1/3
  labels = tf.constant([0, 0, 0, 0, 0, 0, 2, 2, 2], shape=(1, 3, 3, 1))

  prediction_cat = tf.constant([0, 0, 0, 0, 0, 0, 1, 1, 1], shape=(1, 3, 3, 1))
  prediction_one_hot = tf.squeeze(tf.keras.backend.one_hot(prediction_cat, 2))
  weighted_ignorant_accuracy.update_state(labels, prediction_one_hot)
  acc_correct = weighted_ignorant_accuracy.result().numpy()
  weighted_ignorant_accuracy.reset_states()
  assert acc_correct == 1

  prediction_wrong = tf.squeeze(
      tf.keras.backend.one_hot(
          tf.constant([1, 1, 1, 1, 1, 1, 0, 0, 0], shape=(1, 3, 3, 1)), 2))
  weighted_ignorant_accuracy.update_state(labels, prediction_wrong)
  acc_wrong = weighted_ignorant_accuracy.result().numpy()
  weighted_ignorant_accuracy.reset_states()
  assert acc_wrong == 0

  prediction_minority_correct = tf.squeeze(
      tf.keras.backend.one_hot(
          tf.constant([1, 1, 1, 1, 1, 1, 0, 0, 1], shape=(1, 3, 3, 1)), 2))
  weighted_ignorant_accuracy.update_state(labels, prediction_minority_correct)
  acc_minority_correct = weighted_ignorant_accuracy.result().numpy()
  weighted_ignorant_accuracy.reset_states()

  prediction_majority_correct = tf.squeeze(
      tf.keras.backend.one_hot(
          tf.constant([0, 1, 1, 1, 1, 1, 0, 0, 0], shape=(1, 3, 3, 1)), 2))
  weighted_ignorant_accuracy.update_state(labels, prediction_majority_correct)
  acc_majority_correct = weighted_ignorant_accuracy.result().numpy()
  weighted_ignorant_accuracy.reset_states()

  # Predicting the correct label for a minority class should give higher accuracy
  assert acc_majority_correct < acc_minority_correct


def test_ignorant_accuracy():
  """ Test ignorant accuracy with random values """

  ignorant_accuracy = IgnorantAccuracyMetric()
  weighted_ignorant_accuracy = IgnorantBalancedAccuracyMetric()

  # Generate random labels and predictions
  labels = tf.random.uniform(shape=[4, 120, 240, 1],
                             minval=0,
                             maxval=3,
                             dtype=tf.int64)

  prediction = tf.random.uniform(shape=[4, 120, 240, 2], dtype=tf.float64)

  # accuracy for this prediction

  ignorant_accuracy.update_state(labels, prediction)
  random_acc = ignorant_accuracy.result().numpy()
  ignorant_accuracy.reset_states()

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

  ignorant_accuracy.update_state(labels, new_prediction_2)
  random_acc_preturbed = ignorant_accuracy.result().numpy()
  ignorant_accuracy.reset_states()
  assert random_acc == random_acc_preturbed

  true_prediction = tf.stack(
      [labels_as_tensor[..., 0], labels_as_tensor[..., 2]], axis=-1)
  ignorant_accuracy.update_state(labels, true_prediction)
  true_acc = ignorant_accuracy.result().numpy()
  ignorant_accuracy.reset_states()
  assert true_acc == 1

  wrong_prediction = tf.stack(
      [labels_as_tensor[..., 2], labels_as_tensor[..., 0]], axis=-1)
  ignorant_accuracy.update_state(labels, wrong_prediction)
  wrong_acc = ignorant_accuracy.result().numpy()
  ignorant_accuracy.reset_states()
  assert wrong_acc == 0


def test_weighted_ignorant_accuracy():
  """ Test weighted accuracy with random values """

  ignorant_accuracy = IgnorantAccuracyMetric()
  weighted_ignorant_accuracy = IgnorantBalancedAccuracyMetric()

  # Generate random labels and predictions
  labels = tf.random.uniform(shape=[4, 120, 240, 1],
                             minval=0,
                             maxval=3,
                             dtype=tf.int64)

  prediction = tf.random.uniform(shape=[4, 120, 240, 2], dtype=tf.float64)

  # accuracy for this prediction

  weighted_ignorant_accuracy.update_state(labels, prediction)
  random_acc = weighted_ignorant_accuracy.result().numpy()
  weighted_ignorant_accuracy.reset_states()

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

  weighted_ignorant_accuracy.update_state(labels, new_prediction_2)
  random_acc_preturbed = weighted_ignorant_accuracy.result().numpy()
  weighted_ignorant_accuracy.reset_states()

  assert random_acc == random_acc_preturbed

  true_prediction = tf.stack(
      [labels_as_tensor[..., 0], labels_as_tensor[..., 2]], axis=-1)
  weighted_ignorant_accuracy.update_state(labels, true_prediction)
  true_acc = weighted_ignorant_accuracy.result().numpy()
  weighted_ignorant_accuracy.reset_states()
  assert true_acc == 1

  wrong_prediction = tf.stack(
      [labels_as_tensor[..., 2], labels_as_tensor[..., 0]], axis=-1)
  weighted_ignorant_accuracy.update_state(labels, wrong_prediction)
  wrong_acc = weighted_ignorant_accuracy.result().numpy()
  weighted_ignorant_accuracy.reset_states()
  assert wrong_acc == 0

  # Generate unbalanced labels
  labels_unbalanced = tf.where(tf.equal(labels, 1), tf.zeros_like(labels),
                               labels)
  unbalanced_labels_acc = weighted_ignorant_accuracy(labels_unbalanced,
                                                     new_prediction_2).numpy()
  assert unbalanced_labels_acc < random_acc_preturbed
  prediction_all_zeros = tf.where(tf.equal(labels, 1), tf.zeros_like(labels),
                                  labels)
  prediction_all_zeros = tf.where(tf.equal(prediction_all_zeros, 1),
                                  tf.zeros_like(prediction_all_zeros),
                                  prediction_all_zeros)

  weighted_ignorant_accuracy.update_state(labels_unbalanced,
                                          prediction_all_zeros)
  acc_all_zeros_predict_weighted = weighted_ignorant_accuracy.result().numpy()
  weighted_ignorant_accuracy.reset_states()
  ignorant_accuracy.update_state(labels_unbalanced, prediction_all_zeros)
  acc_all_zeros_predict_unweighted = ignorant_accuracy.result().numpy()
  ignorant_accuracy.reset_states()

  assert acc_all_zeros_predict_weighted < acc_all_zeros_predict_unweighted

  prediction_all_twos = tf.where(tf.equal(labels, 1), 2 * tf.ones_like(labels),
                                 labels)
  prediction_all_twos = tf.where(tf.equal(prediction_all_twos, 1),
                                 2 * tf.ones_like(prediction_all_twos),
                                 prediction_all_twos)

  weighted_ignorant_accuracy.update_state(labels_unbalanced,
                                          prediction_all_twos)
  acc_all_twos_predict_weighted = weighted_ignorant_accuracy.result().numpy()
  weighted_ignorant_accuracy.reset_states()

  ignorant_accuracy.update_state(labels_unbalanced, prediction_all_twos)
  acc_all_twos_predict_unweighted = ignorant_accuracy.result().numpy()
  ignorant_accuracy.reset_states()

  assert acc_all_twos_predict_weighted < acc_all_twos_predict_unweighted
