import argparse
import bisect
import incense
from incense import ExperimentLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pvectorc
import pyrsistent
from tensorflow import keras
import yaml as yml
import zipfile

from bfseg.utils.datasets import load_data
from bfseg.utils.evaluation import evaluate_model
from bfseg.utils.models import create_model



class LogExperiment:

  def __init__(self, experiment_id, save_folder, save_output=False):
    loader = ExperimentLoader(mongo_uri=MONGO_URI, db_name='bfseg')

    # Load the experiment.
    self._experiment_id = experiment_id
    self._experiment = loader.find_by_id(self._experiment_id)

    assert (self._experiment.to_dict()['status']
            not in ["FAILED"]), f"Experiment {self._experiment_id} failed."

    # Set up folders.
    save_folder = os.path.abspath(save_folder)
    self._save_folder_models = os.path.join(save_folder, 'models')
    self._save_folder_plots = os.path.join(save_folder,
                                           f'plots/{self._experiment_id}')
    self._save_folder_logs = os.path.join(save_folder,
                                          f'logs/{self._experiment_id}')
    self._save_folder_evaluate = os.path.join(
        save_folder, f'evaluate/{self._experiment_id}')

    folders = [
        self._save_folder_models, self._save_folder_plots,
        self._save_folder_logs, self._save_folder_evaluate
    ]
    for folder in folders:
      if (not os.path.isdir(folder)):
        os.makedirs(folder)

    # Set the epochs for which to log the results (if reached)
    self._epochs_to_save = [100, "final"]

    # Set the test datasets on which to optionally evaluated the pretrained
    # models.
    self._datasets_names_to_evaluate = [
        "BfsegValidationLabeled", "OfficeRumlangValidationLabeled"
    ]
    self._datasets_scenes_to_evaluate = ["CLA", "RUMLANG"]

    self._find_splits_to_log()

    # Save the experiment configuration to file.
    self.save_config_file()

    # Save the plots containing the metrics.
    self.save_plots()

  @staticmethod
  def _recursive_transform_pmap(pmap):
    pmap = dict(pmap)
    for key, value in pmap.items():
      if (isinstance(value, pvectorc.PVector)):
        pmap[key] = LogExperiment._recursive_transform_pvector(value)
      elif (isinstance(value, pyrsistent._pmap.PMap)):
        pmap[key] = LogExperiment._recursive_transform_pmap(value)

    return pmap

  @staticmethod
  def _recursive_transform_pvector(pvector):
    pvector = list(pvector)
    for idx, value in enumerate(pvector):
      if (isinstance(value, pvectorc.PVector)):
        pvector[idx] = LogExperiment._recursive_transform_pvector(value)
      elif (isinstance(value, pyrsistent._pmap.PMap)):
        pvector[idx] = LogExperiment._recursive_transform_pmap(value)

    return pvector

  def save_config_file(self):
    with open(os.path.join(self._save_folder_plots, "config.txt"), "w") as f:
      f.write("Experiment config:\n")
      for key in sorted(self._experiment.config.keys()):
        if (key[-7:] != "_params"):
          continue
        f.write(f"- {key}\n")
        for subkey in sorted(self._experiment.config[key].keys()):
          value = self._experiment.config[key][subkey]
          # If possible, sort also the value (e.g., if it has keys).
          if (hasattr(value, "items")):
            value = dict(sorted(value.items()))
          f.write(f"  - {subkey}: {value}\n")
    # Save the config file also as yaml file.
    with open(os.path.join(self._save_folder_plots, "config.yml"), "w") as f:
      valid_keys = [
          key for key in self._experiment.config.keys() if key[-7:] == "_params"
      ]
      valid_params = {}
      for key in valid_keys:
        value = self._experiment.config[key]
        if (isinstance(value, pvectorc.PVector)):
          value = LogExperiment._recursive_transform_pvector(value)
        elif (isinstance(value, pyrsistent._pmap.PMap)):
          value = LogExperiment._recursive_transform_pmap(value)
        valid_params[key] = value
      yml.dump(valid_params, f)

  def save_model(self, epoch_to_save):
    assert (isinstance(epoch_to_save, int) or epoch_to_save.isnumeric() or
            epoch_to_save == "final")
    artifact_name = f'model_epoch_{epoch_to_save}.zip'

    try:
      complete_model_path = os.path.join(
          self._save_folder_models, f"{self._experiment_id}_{artifact_name}")
      if (os.path.isfile(complete_model_path)):
        print(
            f"Skipping saving of model '{complete_model_path}' because already "
            "existing.")
      else:
        self._experiment.artifacts[artifact_name].save(
            to_dir=self._save_folder_models)
        print(f"Saved model '{complete_model_path}'.")
    except KeyError:
      print(f"Experiment {self._experiment_id} does not have artifact "
            f"{artifact_name} (yet).")
      complete_model_path = None

    return complete_model_path, artifact_name

  def save_output(self):
    with open(os.path.join(self._save_folder_logs, "output_to_screen.txt"),
              "w") as f:
      f.write(self._experiment.to_dict()['captured_out'])

  def evaluate(self, epochs_to_evaluate, datasets_names_to_evaluate,
               datasets_scenes_to_evaluate):
    r"""Evaluates the model with the checkpoint(s) from the given epoch(s) on
    the given test dataset(s).

    Args:
      epochs_to_evaluate (int/str or list of int/str): Epoch(s) of which the
        corresponding saved model should be used to perform evaluation.
      datasets_names_to_evaluate (str or list of str): Names of the dataset(s)
        on which the model(s) should be evaluated.
      datasets_scenes_to_evaluate (str or list of str): Scenes of the dataset(s)
        on which the model(s) should be evaluated.
    
    Returns:
      accuracies (dict): Accuracies, indexed by the concatenation of the dataset
        name and scene and by the epoch number.
      mean_ious (dict): Mean IoUs, indexed by the concatenation of the dataset
        name and scene and by the epoch number.
    """
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
    assert (len(datasets_names_to_evaluate) == len(datasets_scenes_to_evaluate))

    # Create the model.
    encoder, full_model = create_model(model_name="fast_scnn",
                                       freeze_encoder=False,
                                       freeze_whole_model=False,
                                       normalization_type="group",
                                       image_h=480,
                                       image_w=640)
    model = keras.Model(inputs=full_model.input,
                        outputs=[encoder.output, full_model.output])

    accuracies = {}
    mean_ious = {}

    for test_dataset_name, test_dataset_scene in zip(
        datasets_names_to_evaluate, datasets_scenes_to_evaluate):
      curr_dataset_and_scene = f"{test_dataset_name}_{test_dataset_scene}"
      accuracies[curr_dataset_and_scene] = {}
      mean_ious[curr_dataset_and_scene] = {}
      # Skip re-evaluating if evaluation was already performed.
      all_output_evaluation_filenames = [
          os.path.join(
              self._save_folder_evaluate,
              f"{test_dataset_name}_{test_dataset_scene}_epoch_{epoch}.yml")
          for epoch in epochs_to_evaluate
      ]
      epochs_to_evaluate_for_curr_ds = set()
      for epoch, output_evaluation_filename in zip(
              epochs_to_evaluate, all_output_evaluation_filenames):
        if (os.path.exists(output_evaluation_filename)):
          # Load the precomputed accuracies.
          with open(output_evaluation_filename, 'r') as f:
            evaluation_metrics = yml.load(f, Loader=yml.FullLoader)
          accuracies[curr_dataset_and_scene][epoch] = evaluation_metrics[
              'accuracy']
          mean_ious[curr_dataset_and_scene][epoch] = evaluation_metrics[
              'mean_iou']
        else:
          epochs_to_evaluate_for_curr_ds.add(epoch)

      # No evaluation needs to be performed.
      if (len(epochs_to_evaluate_for_curr_ds) == 0):
        print(
            f"Skipping evaluation of model from epochs {epochs_to_evaluate} on "
            f"dataset {test_dataset_name}, scene {test_dataset_scene}, because "
            f"already found at '{all_output_evaluation_filenames}'.")
        continue

      # Load test dataset.
      test_dataset = load_data(dataset_name=test_dataset_name,
                               scene_type=test_dataset_scene,
                               fraction=None,
                               batch_size=8,
                               shuffle_data=False)

      for epoch, output_evaluation_filename in zip(
              epochs_to_evaluate, all_output_evaluation_filenames):
        if (not epoch in epochs_to_evaluate_for_curr_ds):
          print(f"Skipping evaluation of model from epoch {epoch} on dataset "
                f"{test_dataset_name}, scene {test_dataset_scene}, because "
                f"already found at '{output_evaluation_filename}'.")
          continue

        # Retrieve the required model.
        model_path, artifact_name = self.save_model(epoch_to_save=epoch)
        if (model_path is None):
          print(f"Cannot evaluate current experiment at epoch {epoch}, because "
                "the corresponding model does not exist.")
          continue
        # Extract the pretrained model from the archive, if necessary.
        weights_file_name = artifact_name.split('.zip')[0] + ".h5"
        output_filename = f"{self._experiment_id}_{weights_file_name}"
        extracted_model_path = os.path.join(self._save_folder_models,
                                            output_filename)
        if (not os.path.isfile(extracted_model_path)):
          assert (model_path[-4:] == ".zip")
          with zipfile.ZipFile(model_path, 'r') as zip_helper:
            assert (zip_helper.namelist() == [weights_file_name])
            # Rename the pretrained model so as to include the experiment ID.
            f = zip_helper.open(weights_file_name)
            file_content = f.read()
            f = open(extracted_model_path, 'wb')
            f.write(file_content)
            f.close()
        # Evaluate the pretrained model on the given dataset.
        accuracy, mean_iou = evaluate_model(model=model,
                                            test_dataset=test_dataset,
                                            pretrained_dir=extracted_model_path)
        accuracies[curr_dataset_and_scene][epoch] = accuracy
        mean_ious[curr_dataset_and_scene][epoch] = mean_iou
        # Write the result to file.
        with open(output_evaluation_filename, 'w') as f:
          yml.dump({'accuracy': accuracy, 'mean_iou': mean_iou}, f)
        print(f"Saved evaluation of model from epoch {epoch} on dataset "
              f"{test_dataset_name}, scene {test_dataset_scene} at "
              f"'{output_evaluation_filename}'.")

    return accuracies, mean_ious

  def _find_splits_to_log(self):
    self._splits_to_log = []
    self._metrics_to_log = ["accuracy", "mean_iou"]  #, "loss"]:

    for split in [
        "train", "train_no_replay", "val", "test", "NyuDepthV2Labeled_None",
        "BfsegCLAMeshdistLabels_None"
    ]:
      all_metrics_found_for_split = True
      for metric in self._metrics_to_log:
        if (f'{split}_{metric}' not in self._experiment.metrics):
          all_metrics_found_for_split = False
          break
      if (all_metrics_found_for_split):
        self._splits_to_log.append(split)

  def save_results(self, evaluate_on_validation):

    def write_result(split, metric, epoch, out_file, value=None):

      try:
        if (epoch == "final"):
          epoch_number = self._num_epochs - 1
          epoch_text = [f"final epoch", f" (ep. {epoch_number})"]
        else:
          epoch_number = epoch
          epoch_text = [f"epoch {epoch_number}", ""]

        if (value is None):
          value = self._experiment.metrics[f'{split}_{metric}'].values[
              epoch_number]
        value_text = "{:.4f}".format(value) + f"{epoch_text[1]}"

        out_file.write(f"- {split} {metric} @ {epoch_text[0]}: " + value_text +
                       "\n")

        return value_text
      except IndexError:
        return None

    if (evaluate_on_validation):
      val_accuracies, val_mean_ious = self.evaluate(
          epochs_to_evaluate=self._epochs_to_save,
          datasets_names_to_evaluate=self._datasets_names_to_evaluate,
          datasets_scenes_to_evaluate=self._datasets_scenes_to_evaluate)

    def get_updated_cell_text(cell_text, split, metric, epoch, f, value=None):
      value_text = write_result(split=split,
                                metric=metric,
                                epoch=epoch,
                                out_file=f,
                                value=value)
      if (value_text is not None):
        if (len(cell_text) == 0):
          cell_text = value_text
        else:
          cell_text += f"\n{value_text}"

      return cell_text

    with open(os.path.join(self._save_folder_plots, "results.txt"), "w") as f:
      full_text = [[]]
      column_headers = []
      for metric in self._metrics_to_log:
        for split in self._splits_to_log:
          cell_text = ""
          column_headers.append(f"{metric} {split}")
          for epoch in self._epochs_to_save:
            cell_text = get_updated_cell_text(cell_text=cell_text,
                                              split=split,
                                              metric=metric,
                                              epoch=epoch,
                                              f=f)
          full_text[0].append(cell_text)
        # Log also the metrics from the optional evaluation.
        if ((metric in ["accuracy", "mean_iou"]) and evaluate_on_validation):
          if (metric == "accuracy"):
            for split in val_accuracies.keys():
              cell_text = ""
              column_headers.append(f"{metric} {split}")
              assert (set(self._epochs_to_save) == set(
                  val_accuracies[split].keys()))
              for epoch in self._epochs_to_save:
                value = val_accuracies[split][epoch]
                cell_text = get_updated_cell_text(cell_text=cell_text,
                                                  split=split,
                                                  metric=metric,
                                                  epoch=epoch,
                                                  f=f,
                                                  value=value)
              full_text[0].append(cell_text)
          else:
            for split in val_mean_ious.keys():
              cell_text = ""
              column_headers.append(f"{metric} {split}")
              assert (set(self._epochs_to_save) == set(
                  val_mean_ious[split].keys()))
              for epoch in self._epochs_to_save:
                value = val_mean_ious[split][epoch]
                cell_text = get_updated_cell_text(cell_text=cell_text,
                                                  split=split,
                                                  metric=metric,
                                                  epoch=epoch,
                                                  f=f,
                                                  value=value)
              full_text[0].append(cell_text)

      # Also write the results as an excel file, for easier logging to
      # Spreadsheet.
      df = pd.DataFrame(full_text,
                        index=[f"{self._experiment_id}"],
                        columns=column_headers)
      df.to_excel(os.path.join(self._save_folder_plots, "excel_results.ods"))

  def save_plots(self):
    split_with_metric = {split: [] for split in self._splits_to_log}
    for full_metric_name in self._experiment.metrics.keys():
      if (full_metric_name == "lr"):
        # Learning rate is handled separately.
        continue
      # Try first with the special cases.
      prefix = None
      for special_split_name in [
          "train_no_replay_", "NyuDepthV2Labeled_None_",
          "BfsegCLAMeshdistLabels_None_"
      ]:
        split_metric_name = full_metric_name.split(special_split_name,
                                                   maxsplit=1)
        if (len(split_metric_name) != 1):
          prefix = special_split_name[:-1]
          metric_name = split_metric_name[-1]
          break

      if (prefix is None):
        # Case `train`, `test`, or `val`,
        prefix, metric_name = full_metric_name.split("_", maxsplit=1)
        if (not prefix in split_with_metric.keys()):
          raise KeyError(f"Metric {full_metric_name} was not recognized.")

      if (prefix in split_with_metric.keys()):
        bisect.insort(split_with_metric[prefix], metric_name)
    len_first_element = len(split_with_metric[list(
        split_with_metric.keys())[0]])
    # Check that all split types (e.g., 'train') have the same metrics.
    split_types = list(split_with_metric.keys())
    metric_names_per_type = list(split_with_metric.values())
    assert (metric_names_per_type.count(
        metric_names_per_type[0]) == len(metric_names_per_type))
    metric_names = metric_names_per_type[0]

    metrics_to_log = [[
        f"{split_type}_{metric_name}" for split_type in split_types
    ] for metric_name in metric_names]

    self._num_epochs = len(self._experiment.metrics[list(
        self._experiment.metrics.keys())[0]])

    # If present, log learning rate.
    if ("lr" in self._experiment.metrics.keys()):
      # Learning rate is handled separately.
      metrics_to_log.append(["lr"])

    for group_idx, metric_group in enumerate(metrics_to_log):
      fig = plt.figure(num=f'Run {self._experiment_id}, fig {group_idx}',
                       figsize=(12, 10))
      ax = fig.add_subplot(1, 1, 1)

      for metric in metric_group:
        self._experiment.metrics[metric].index += 1
        self._experiment.metrics[metric].plot(ax=ax)

      if (self._num_epochs + 1 >= 5):
        major_ticks = np.linspace(1, self._num_epochs, 5, dtype=int)
      else:
        major_ticks = self._experiment.metrics[metric].index

      minor_ticks = np.arange(1, self._num_epochs + 1, 1)
      ax.set_xticks(minor_ticks, minor=True)
      ax.grid(which='minor', alpha=0.2)

      ax.set_xticks(major_ticks)
      ax.grid(which='major', alpha=0.5)

      ax.legend()

      # Save the plot to file.
      plt.savefig(os.path.join(self._save_folder_plots,
                               f"plot_{group_idx}.png"))


if (__name__ == "__main__"):
  parser = argparse.ArgumentParser()
  parser.add_argument("--id",
                      type=int,
                      help="ID of the experiment to log.",
                      required=True)
  parser.add_argument("--save_folder",
                      type=str,
                      help="Path where the logs should be saved.",
                      required=True)
  parser.add_argument(
      "--model_to_save",
      help=
      "If specified, epoch of the model to save (a valid epoch number or the "
      "string 'final', for the final model.",
      default=None)
  parser.add_argument(
      "--save_output",
      action="store_true",
      help="Whether or not to save the output to screen of the experiment.")
  parser.add_argument(
      "--save_results",
      action="store_true",
      help="Whether or not to save a summary of the results of the experiment.")
  parser.add_argument(
      "--evaluate",
      action="store_true",
      help="Whether or not to evaluate selected pretrained models on selected "
      "validation sets.")
  args = parser.parse_args()
  experiment_id = args.id
  save_folder = args.save_folder
  save_output = args.save_output
  save_results = args.save_results
  evaluate = args.evaluate
  model_to_save = args.model_to_save

  experiment_logger = LogExperiment(experiment_id=experiment_id,
                                    save_folder=save_folder)

  # Optionally save a model to file.
  if (model_to_save is not None):
    experiment_logger.save_model(epoch_to_save=model_to_save)
  # Optionally save the output to file.
  if (save_output):
    experiment_logger.save_output()
  # Optionally save a summary of the results to file.
  if (save_results):
    experiment_logger.save_results(evaluate_on_validation=evaluate)
