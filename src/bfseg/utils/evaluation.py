import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import yaml as yml

from bfseg.utils.datasets import load_data
from bfseg.utils.metrics import (IgnorantBalancedMeanIoU, IgnorantMeanIoU,
                                 IgnorantBalancedAccuracyMetric,
                                 IgnorantAccuracyMetric)
from bfseg.utils.models import create_model


def oneMetricIteration(metric, label, pred):
  """ Helper function to get the result from one prediction """
  metric.update_state(label, pred)
  res = metric.result().numpy()
  metric.reset_states()
  return res


def scoreAndPlotPredictions(imageCallback,
                            test_ds,
                            num_images,
                            plot=True,
                            outFolder=None,
                            tag="",
                            exportPredictions=False):
  """
  Calculates different metrices for the validation set provided by the dataloader.
  Also plots predictions if plot = True

  Args:
      imageCallback: lambda function that takes a batch of images and returns the prediction
      dataLoader: data.meshdist.dataLoader
      plot: Flag whether to plot the results or not
      outFolder: Where to store the prediction
      tag: tag appended to the prediction results
      exportPredictions: Flag whether or not to export predicted images

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

  FNM_valid = tf.keras.metrics.FalseNegatives()
  TNM_valid = tf.keras.metrics.TrueNegatives()
  TPM_valid = tf.keras.metrics.TruePositives()
  iam_valid = IgnorantAccuracyMetric()
  ibm_valid = IgnorantBalancedAccuracyMetric()

  MIOUM_B_valid = IgnorantBalancedMeanIoU()
  MIOUM_valid = IgnorantMeanIoU()

  # count images
  cnt = 0
  for test_img, test_label in test_ds.take(-1):
    pred = imageCallback(test_img)
    if cnt >= num_images:
      break

    for i in range(pred.shape[0]):
      if cnt >= num_images:
        break
      cnt += 1

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
      ibm_value = oneMetricIteration(ibm, test_label[i, ...], pred[i, ...])
      iam_valid.update_state(test_label[i, ...], pred[i, ...])
      ibm_valid.update_state(test_label[i, ...], pred[i, ...])

      mIoU = oneMetricIteration(MIOUM, test_label[i, ...], pred[i, ...])
      mIoU_B = oneMetricIteration(MIOUM_B, test_label[i, ...], pred[i, ...])

      MIOUM_valid.update_state(test_label[i, ...], pred[i, ...])
      MIOUM_B_valid.update_state(test_label[i, ...], pred[i, ...])

      # plot results using matplotlib
      if plot:
        plt.subplot(num_images // 5, 5, cnt)
        plt.imshow(tf.argmax(pred[i], axis=-1))
        plt.imshow(test_img[i], alpha=0.7)
        plt.title(
            f"mIoU: {mIoU:.4f}, mIoU_B: {mIoU_B:.4f}\nIAM: {iam_value:.4f}, IBM: {ibm_value:.4f}\nTPR: {TP / (FP + FN):.4f}, TNR {TN / (TN + FP):.4f} "
        )

      # Export results as csv
      if outFolder is not None:
        img_name = tag + "_" + str(cnt).zfill(3) + ".png"
        # Create csv entry for each image
        with open(os.path.join(outFolder, "results_one_by_one_" + tag + ".csv"),
                  "a+") as f:
          f.write(
              f"{img_name},{iam_value:.4f},{ibm_value:.4f},{mIoU:.4f},{mIoU_B:.4f}\n"
          )

        # Export predicted images if requested
        if exportPredictions:
          imgs_folder = os.path.join(outFolder, "imgs")
          if not os.path.exists(imgs_folder):
            os.mkdir(imgs_folder)

          Image.fromarray(np.uint8(tf.argmax(pred[i], axis=-1)), 'L').save(
              os.path.join(imgs_folder, "pred_" + img_name))
          Image.fromarray(np.uint8(np.squeeze(test_label[i])),
                          'L').save(os.path.join(imgs_folder, "gt_" + img_name))
          Image.fromarray(np.uint8(test_img[i] * 255)).save(
              os.path.join(imgs_folder, "img_" + img_name))

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

  # Export results.csv containing accuracy and iou information
  if outFolder is not None:
    with open(os.path.join(outFolder, "results.csv"), "a+") as f:
      f.write(
          f"{tag},{iam_valid_value:.4f},{ibm_valid_value:.4f},{MIOUM_valid_value:.4f},{MIOUM_B_valid_value:.4f}\n"
      )


def evaluate_model(model, test_dataset, pretrained_dir=None):
  r"""Evaluates a model on a given test dataset.

  Args:
    model (tensorflow.keras.Model): Model to evaluate.
    test_dataset (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Test
      dataset on which to evaluate the CL model.
    pretrained_dir (str): If not None, path from which the weights of a
      pretrained model will be loaded.

  Returns:
    accuracy (float): Accuracy of the given model over the given test dataset.
    mean_iou (float): Mean IoU of the given model over the given test dataset.
  """
  # Optionally load weights.
  if (pretrained_dir is not None):
    model.load_weights(pretrained_dir)
  accuracy_tracker = tf.keras.metrics.Accuracy(name='accuracy',
                                               dtype=tf.float32)
  miou_tracker = tf.keras.metrics.MeanIoU(name='mean_iou', num_classes=2)

  accuracy_tracker.reset_states()
  miou_tracker.reset_states()
  for sample in test_dataset:
    if (len(sample) == 3):
      x, y, mask = sample
    else:
      assert (len(sample) == 2)
      x, y = sample
      mask = tf.ones(shape=x.shape[:-1])
    [_, pred_y] = model(x, training=False)
    y_masked = tf.boolean_mask(y, mask)
    pred_y = tf.keras.backend.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, mask)
    accuracy_tracker.update_state(y_masked, pred_y_masked)
    miou_tracker.update_state(y_masked, pred_y_masked)

  accuracy = accuracy_tracker.result().numpy().item()
  mean_iou = miou_tracker.result().numpy().item()

  return accuracy, mean_iou


def evaluate_model_multiple_epochs_and_datasets(pretrained_dirs,
                                                epochs_to_evaluate,
                                                datasets_names_to_evaluate,
                                                datasets_scenes_to_evaluate,
                                                save_folder):
  r"""Evaluates a model with the checkpoint(s) from the given epoch(s) on the
  given test dataset(s).

  Args:
    pretrained_dirs (str or list of str): Path to pretrained models that should
      be used to perform the evaluation.
    epochs_to_evaluate (int/str or list of int/str): Epoch(s) at which the
      corresponding model used to perform evaluation was saved.
    datasets_names_to_evaluate (str or list of str): Names of the dataset(s) on
      which the model(s) should be evaluated.
    datasets_scenes_to_evaluate (str or list of str): Scenes of the dataset(s)
      which the model(s) should be evaluated.
    save_folder (str): Folder where the results of the evalutions should be
      saved.
  
  Returns:
    accuracies (dict): Accuracies, indexed by the concatenation of the dataset
      name and scene and by the epoch number.
    mean_ious (dict): Mean IoUs, indexed by the concatenation of the dataset
      name and scene and by the epoch number.
  """
  if (isinstance(pretrained_dirs, str)):
    pretrained_dirs = [pretrained_dirs]
  else:
    assert (isinstance(pretrained_dirs, list))
  if (isinstance(epochs_to_evaluate, int)):
    epochs_to_evaluate = [epochs_to_evaluate]
  else:
    assert (isinstance(epochs_to_evaluate, list))
  if (isinstance(datasets_names_to_evaluate, str)):
    datasets_names_to_evaluate = [datasets_names_to_evaluate]
  else:
    assert (isinstance(datasets_names_to_evaluate, list))
  if (isinstance(datasets_scenes_to_evaluate, str)):
    datasets_scenes_to_evaluate = [datasets_scenes_to_evaluate]
  else:
    assert (isinstance(datasets_scenes_to_evaluate, list))
  assert (len(pretrained_dirs) == len(epochs_to_evaluate))
  assert (len(datasets_names_to_evaluate) == len(datasets_scenes_to_evaluate))

  # Create the model.
  encoder, full_model = create_model(model_name="fast_scnn",
                                     freeze_encoder=False,
                                     freeze_whole_model=False,
                                     normalization_type="group",
                                     image_h=480,
                                     image_w=640)
  model = tf.keras.Model(inputs=full_model.input,
                         outputs=[encoder.output, full_model.output])

  accuracies = {}
  mean_ious = {}

  for test_dataset_name, test_dataset_scene in zip(datasets_names_to_evaluate,
                                                   datasets_scenes_to_evaluate):
    curr_dataset_and_scene = f"{test_dataset_name}_{test_dataset_scene}"
    accuracies[curr_dataset_and_scene] = {}
    mean_ious[curr_dataset_and_scene] = {}
    # Skip re-evaluating if evaluation was already performed.
    all_output_evaluation_filenames = [
        os.path.join(
            save_folder,
            f"{test_dataset_name}_{test_dataset_scene}_epoch_{epoch}.yml")
        for epoch in epochs_to_evaluate
    ]
    epochs_to_evaluate_for_curr_ds = set()
    for epoch, output_evaluation_filename, pretrained_dir in zip(
        epochs_to_evaluate, all_output_evaluation_filenames, pretrained_dirs):
      if (os.path.exists(output_evaluation_filename)):
        # Load the precomputed accuracies.
        with open(output_evaluation_filename, 'r') as f:
          evaluation_metrics = yml.load(f, Loader=yml.FullLoader)
        # Check that also the pretrained model used matches.
        if (evaluation_metrics['pretrained_dir'] == pretrained_dir):
          accuracies[curr_dataset_and_scene][epoch] = evaluation_metrics[
              'accuracy']
          mean_ious[curr_dataset_and_scene][epoch] = evaluation_metrics[
              'mean_iou']
        else:
          epochs_to_evaluate_for_curr_ds.add(epoch)
      else:
        epochs_to_evaluate_for_curr_ds.add(epoch)

    # No evaluation needs to be performed.
    if (len(epochs_to_evaluate_for_curr_ds) == 0):
      print(f"Skipping evaluation of model from epochs {epochs_to_evaluate} on "
            f"dataset {test_dataset_name}, scene {test_dataset_scene}, because "
            f"already found at '{all_output_evaluation_filenames}'.")
      continue

    # Load test dataset.
    test_dataset = load_data(dataset_name=test_dataset_name,
                             scene_type=test_dataset_scene,
                             fraction=None,
                             batch_size=8,
                             shuffle_data=False)

    for epoch, output_evaluation_filename, pretrained_dir in zip(
        epochs_to_evaluate, all_output_evaluation_filenames, pretrained_dirs):
      if (not epoch in epochs_to_evaluate_for_curr_ds):
        print(f"Skipping evaluation of model from epoch {epoch} on dataset "
              f"{test_dataset_name}, scene {test_dataset_scene}, because "
              f"already found at '{output_evaluation_filename}'.")
        continue

      # Evaluate the pretrained model on the given dataset.
      accuracy, mean_iou = evaluate_model(model=model,
                                          test_dataset=test_dataset,
                                          pretrained_dir=pretrained_dir)
      accuracies[curr_dataset_and_scene][epoch] = accuracy
      mean_ious[curr_dataset_and_scene][epoch] = mean_iou
      # Write the result to file.
      with open(output_evaluation_filename, 'w') as f:
        yml.dump(
            {
                'accuracy': accuracy,
                'mean_iou': mean_iou,
                'pretrained_dir': pretrained_dir
            }, f)
      print(f"Saved evaluation of model from epoch {epoch} on dataset "
            f"{test_dataset_name}, scene {test_dataset_scene} at "
            f"'{output_evaluation_filename}'.")

  return accuracies, mean_ious
