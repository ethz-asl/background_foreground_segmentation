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
import yaml as yml

MONGO_URI = INSERT YOUR MONGO URI HERE


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

    folders = [
        self._save_folder_models, self._save_folder_plots,
        self._save_folder_logs
    ]
    for folder in folders:
      if (not os.path.isdir(folder)):
        os.makedirs(folder)

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
      self._experiment.artifacts[artifact_name].save(
          to_dir=self._save_folder_models)
      complete_model_path = os.path.join(
          self._save_folder_models, f"{self._experiment_id}_{artifact_name}")
      print(f"Saved model '{complete_model_path}'.")
    except KeyError:
      print(f"Experiment {self._experiment_id} does not have artifact "
            f"{artifact_name} (yet).")

  def save_output(self):
    with open(os.path.join(self._save_folder_logs, "output_to_screen.txt"),
              "w") as f:
      f.write(self._experiment.to_dict()['captured_out'])

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

  def save_results(self):

    def write_result(split, metric, epoch, out_file):

      try:
        if (epoch == "final"):
          epoch_number = len(
              self._experiment.metrics[f'{split}_{metric}'].values) - 1
          epoch_text = [f"final epoch", f" (ep. {epoch_number})"]
        else:
          epoch_number = epoch
          epoch_text = [f"epoch {epoch_number}", ""]

        value_text = "{:.4f}".format(
            self._experiment.metrics[f'{split}_{metric}'].values[epoch_number]
        ) + f"{epoch_text[1]}"
        out_file.write(f"- {split} {metric} @ {epoch_text[0]}: " + value_text +
                       "\n")

        return value_text
      except IndexError:
        return None

    # Save results at epoch 100 (if reached), and at the final epoch.
    epochs_to_save = [100, "final"]

    with open(os.path.join(self._save_folder_plots, "results.txt"), "w") as f:
      full_text = [[]]
      column_headers = []
      for metric in self._metrics_to_log:
        for split in self._splits_to_log:
          cell_text = ""
          column_headers.append(f"{metric} {split}")
          for epoch in epochs_to_save:
            value_text = write_result(split=split,
                                      metric=metric,
                                      epoch=epoch,
                                      out_file=f)
            if (value_text is not None):
              if (len(cell_text) == 0):
                cell_text = value_text
              else:
                cell_text += f"\n{value_text}"
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

    num_epochs = len(self._experiment.metrics[list(
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

      if (num_epochs + 1 >= 5):
        major_ticks = np.linspace(1, num_epochs, 5, dtype=int)
      else:
        major_ticks = self._experiment.metrics[metric].index

      minor_ticks = np.arange(1, num_epochs + 1, 1)
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
  args = parser.parse_args()
  experiment_id = args.id
  save_folder = args.save_folder
  save_output = args.save_output
  save_results = args.save_results
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
    experiment_logger.save_results()
