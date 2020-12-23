from bfseg.utils.metrics import IgnorantBalancedMeanIoU, IgnorantMeanIoU, IgnorantBalancedAccuracyMetric, IgnorantAccuracyMetric
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def oneMetricIteration(metric, label, pred):
  """ Helper function to get the result from one prediction """
  metric.update_state(label, pred)
  res = metric.result().numpy()
  metric.reset_states()
  return res


def scoreAndPlotPredictions(imageCallback, test_ds, num_images, plot=True, batchSize = 5, outFolder = None, tag = "", exportPredictions = False):
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

  MIOUM_B = IgnorantBalancedMeanIoU()
  MIOUM = IgnorantMeanIoU()

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

  MIOUM_B_valid = IgnorantBalancedMeanIoU()
  MIOUM_valid = IgnorantMeanIoU()

  batches = num_images // batchSize #- 1
  cnt = 0
  for test_img, test_label in test_ds.take(batches):
    pred = imageCallback(test_img)

    for i in range(pred.shape[0]):
      if plot:
        plt.subplot(batches, 5, i + cnt * batches + 1)
        plt.imshow(tf.argmax(pred[i], axis=-1))
        plt.imshow(test_img[i], alpha=0.7)

      # Convert prediction to categorical form
      pred_categorical = tf.argmax(pred[i], axis=-1)

      # True false label needed for True Positive, False Negatives, ....
      label_true_false = test_label[i] > 0

      FN = oneMetricIteration(FNM, label_true_false, pred_categorical)
      FP = oneMetricIteration(FPM, label_true_false, pred_categorical)
      TP = oneMetricIteration(TPM, label_true_false, pred_categorical)
      TN = oneMetricIteration(TNM, label_true_false, pred_categorical)

      FPM_valid.update_state(label_true_false, pred_categorical)
      FNM_valid.update_state(label_true_false, pred_categorical)
      TNM_valid.update_state(label_true_false, pred_categorical)
      TPM_valid.update_state(label_true_false, pred_categorical)

      # Update Accuracy metrics
      iam_value = oneMetricIteration(iam, test_label[i, ...], pred[i, ...])
      ibm_value = oneMetricIteration(iam, test_label[i, ...], pred[i, ...])
      iam_valid.update_state(test_label[i, ...], pred[i, ...])
      ibm_valid.update_state(test_label[i, ...], pred[i, ...])

      mIoU = oneMetricIteration(MIOUM, test_label[i, ...], pred[i, ...])
      mIoU_B = oneMetricIteration(MIOUM_B, test_label[i, ...], pred[i, ...])

      MIOUM_valid.update_state(test_label[i, ...], pred[i, ...])
      MIOUM_B_valid.update_state(test_label[i, ...], pred[i, ...])

      if plot:
        plt.title(
            f"mIoU: {mIoU:.4f}, mIoU_B: {mIoU_B:.4f}\nIAM: {iam_value:.4f}, IBM: {ibm_value:.4f}\nTPR: {TP / (FP + FN):.4f}, TNR {TN / (TN + FP):.4f} "
        )

      if outFolder is not None:
        img_name = tag + "_" + str(i + cnt * batches).zfill(3) + ".png"

        with open(os.path.join(outFolder, "results_one_by_one_" + tag + ".csv"), "a+") as f:
          f.write(f"{img_name},{iam_value:.4f},{ibm_value:.4f},{mIoU:.4f},{mIoU_B:.4f}\n")

        if exportPredictions:
          imgs_folder = os.path.join(outFolder, "imgs")
          if not os.path.exists(imgs_folder):
            os.mkdir(imgs_folder)

          Image.fromarray(np.uint8(tf.argmax(pred[i], axis=-1)), 'L').save(os.path.join(imgs_folder, "pred_" + img_name))
          Image.fromarray(np.uint8(np.squeeze(test_label[i])), 'L').save(os.path.join(imgs_folder, "gt_" + img_name))
          Image.fromarray(np.uint8(test_img[i]*255)).save(os.path.join(imgs_folder, "img_" + img_name))

    cnt += 1

  FP = FPM_valid.result().numpy()
  FN = FNM_valid.result().numpy()
  TN = TNM_valid.result().numpy()
  TP = TPM_valid.result().numpy()

  iam_valid_value = iam_valid.result().numpy()
  MIOUM_valid_value = MIOUM_valid.result().numpy()
  ibm_valid_value = ibm_valid.result().numpy()
  MIOUM_B_valid_value = MIOUM_B_valid.result().numpy()

  print("Accuracy on validation set:", iam_valid_value)
  print("Balanced Accuracy on validation set:", ibm_valid_value)
  print("mIoU on validation set:", MIOUM_valid_value)
  print("Balanced mIoU on validation set:", MIOUM_B_valid_value)

  print("Positive = Foreground, Negative = Background")
  print("TPR on validation set:", TP / (TP + FN))
  print("TNR on validation set:", TN / (TN + FP))
  print("Precision  on validation set:", TP / (TP + FP))

  if outFolder is not None:
    with open(os.path.join(outFolder,"results.csv"), "a+") as f:
      f.write(f"{tag},{iam_valid_value:.4f},{ibm_valid_value:.4f},{MIOUM_valid_value:.4f},{MIOUM_B_valid_value:.4f}\n")
