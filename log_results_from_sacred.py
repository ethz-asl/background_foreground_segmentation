import argparse
import bisect
import incense
from incense import ExperimentLoader
import matplotlib.pyplot as plt
import numpy as np
import os

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

    # Save the experiment configuration to file.
    self.save_config_file()

    # Save the plots containing the metrics.
    self.save_plots()

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

  def save_plots(self):
    metrics_to_log = {'train': [], 'test': [], 'val': []}
    for full_metric_name in self._experiment.metrics.keys():
      prefix, metric_name = full_metric_name.split("_", maxsplit=1)
      if (prefix in metrics_to_log.keys()):
        bisect.insort(metrics_to_log[prefix], metric_name)
    len_first_element = len(metrics_to_log[list(metrics_to_log.keys())[0]])
    # Check that all metric types (e.g., 'train') have the same metrics.
    metric_types = list(metrics_to_log.keys())
    metric_names_per_type = list(metrics_to_log.values())
    assert (metric_names_per_type.count(
        metric_names_per_type[0]) == len(metric_names_per_type))
    metric_names = metric_names_per_type[0]

    metrics_to_log = [[
        f"{metric_type}_{metric_name}" for metric_type in metric_types
    ] for metric_name in metric_names]

    num_epochs = len(self._experiment.metrics[list(
        self._experiment.metrics.keys())[0]])

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
  args = parser.parse_args()
  experiment_id = args.id
  save_folder = args.save_folder
  save_output = args.save_output
  model_to_save = args.model_to_save

  experiment_logger = LogExperiment(experiment_id=experiment_id,
                                    save_folder=save_folder,
                                    save_output=save_output)

  # Optionally save a model to file.
  if (model_to_save is not None):
    experiment_logger.save_model(epoch_to_save=model_to_save)
  # Optionally save the output to file.
  if (save_output):
    experiment_logger.save_output()
