def oneMetricIteration(metric, label, pred):
  """ Helper function to get the result from one prediction """
  metric.update_state(label, pred)
  res = metric.result().numpy()
  metric.reset_states()
  return res


def scoreAndPlotPredictions(imageCallback, dataLoader, plot=True):
  """
  Calculates different metrices for the validation set provided by the dataloader.
  Also plots predictions if plot = True
  Args:
      imageCallback: lambda function that takes a batch of images and returns the prediction
      dataLoader: data.meshdist.dataLoader
      plot: Flag whether to plot the results or not

  Returns: Nothing, currently only prints
  """
  iam = IgnorantAccuracyMetric()
  ibm = IgnorantBalancedAccuracyMetric()

  # object assigned the 0 label is background, assigned the 1 label is foreground.
  # False Pos: background that is interpreted as foregound
  FPM = tf.keras.metrics.FalsePositives(
  )  # object assigned the 0 label is background, assigned the 1 label is foreground.
  # False Neg: Foreground that is interpreted as background
  FNM = tf.keras.metrics.FalseNegatives()
  TNM = tf.keras.metrics.TrueNegatives()
  TPM = tf.keras.metrics.TruePositives()

  MIOUM = tf.keras.metrics.MeanIoU(num_classes=2)

  # Valid metrices accumulate results over whole validation set.
  # Other metrices are only used to calculate results for every image.
  FPM_valid = tf.keras.metrics.FalsePositives(
  )  # object assigned the 0 label is background, assigned the 1 label is foreground.
  # False Neg: Foreground that is interpreted as background
  FNM_valid = tf.keras.metrics.FalseNegatives()
  TNM_valid = tf.keras.metrics.TrueNegatives()
  TPM_valid = tf.keras.metrics.TruePositives()
  iam_valid = IgnorantAccuracyMetric()
  ibm_valid = IgnorantBalancedAccuracyMetric()

  MIOUM_valid = tf.keras.metrics.MeanIoU(num_classes=2)

  train_ds, test_ds = dataLoader.getDataset()
  batches = dataLoader.validationSize // 5 - 1
  cnt = 0
  for test_img, test_label in test_ds.take(batches):
    pred = imageCallback(test_img)

    for i in range(5):
      if plot:
        plt.subplot(batches, 5, i + cnt * batches + 1)
        plt.imshow(tf.argmax(pred[i], axis=-1))
        plt.imshow(test_img[i], alpha=0.7)

      FN = oneMetricIteration(FNM, test_label[i] > 0, tf.argmax(pred[i],
                                                                axis=-1))
      FP = oneMetricIteration(FPM, test_label[i] > 0, tf.argmax(pred[i],
                                                                axis=-1))
      TP = oneMetricIteration(TPM, test_label[i] > 0, tf.argmax(pred[i],
                                                                axis=-1))
      TN = oneMetricIteration(TNM, test_label[i] > 0, tf.argmax(pred[i],
                                                                axis=-1))

      mIoU = oneMetricIteration(MIOUM, test_label[i] > 0,
                                tf.argmax(pred[i], axis=-1))

      FPM_valid.update_state(test_label[i] > 0, tf.argmax(pred[i], axis=-1))
      FNM_valid.update_state(test_label[i] > 0, tf.argmax(pred[i], axis=-1))
      TNM_valid.update_state(test_label[i] > 0, tf.argmax(pred[i], axis=-1))
      TPM_valid.update_state(test_label[i] > 0, tf.argmax(pred[i], axis=-1))

      MIOUM_valid.update_state(test_label[i] > 0, tf.argmax(pred[i], axis=-1))

      # Update Accuracy metrics
      iam_value = oneMetricIteration(iam, test_label[i, ...], pred[i, ...])
      ibm_value = oneMetricIteration(iam, test_label[i, ...], pred[i, ...])
      iam_valid.update_state(test_label[i, ...], pred[i, ...])
      ibm_valid.update_state(test_label[i, ...], pred[i, ...])

      if plot:
        plt.title("IAM:" + str(round(iam_value, 4)) + ", IBM:" +
                  str(round(ibm_value, 4)) + "\n " + "TPR:" +
                  str(round(TP / (FP + FN), 4)) + " TNR:" +
                  str(round(TN / (TN + FP), 4)) + "\n " + " mIoU: " +
                  str(round(mIoU, 4)))

    cnt += 1

  FP = FPM_valid.result().numpy()
  FN = FNM_valid.result().numpy()
  TN = TNM_valid.result().numpy()
  TP = TPM_valid.result().numpy()

  print("Accuracy on validation set:", iam_valid.result().numpy())
  print("mIoU on validation set:", MIOUM_valid.result().numpy())
  print("Balanced Accuracy on validation set:", ibm_valid.result().numpy())
  print("Positive = Foreground, Negative = Background")
  print("TPR on validation set:", TP / (TP + FN))
  print("TNR on validation set:", TN / (TN + FP))
  print("Precision  on validation set:", TP / (TP + FP))
