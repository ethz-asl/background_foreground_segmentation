import sys
sys.path.append("..")
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import datetime
from sacred import Experiment

from bfseg.cl_experiments import BaseSegExperiment
from bfseg.sacred_utils import get_observer
from bfseg.settings import TMPDIR

ex = Experiment()
ex.observers.append(get_observer())


@ex.config
def seg_experiment_default_config():
  r"""Default configuration for base segmentation experiments.
  - Network parameters:
    - architecture (str): Architecture type. Valid values are:
      - "unet": U-Net architecture. Required parameters are:
        - image_w (int): Image width.
        - image_h (int): Image height.
        - backbone (str): Name of the backbone of the U-Net architecture.
  - Training parameters:
    - batch_size (int): Batch size.
    - learning_rate (float): Learning rate.
    - num_training_epochs (int): Number of training epochs.
  - Dataset parameters:
    - test_dataset (str): Name of the test dataset.
    - test_scene (str): Scene type of the test dataset. Valid values are: None,
        "kitchen", "bedroom".
    - train_dataset (str): Name of the training dataset.
    - train_scene (str): Scene type of the training dataset. Valid values are:
        None, "kitchen", "bedroom".
    - validation_percentage (int): Percentage of the training scene to use for
      validation.
  - Logging parameters:
    - exp_name (str): Name of the current experiment.
    - metric_log_frequency (str): Frequency with which the training metrics are
      logged. Valid values are "epoch" (i.e., every epoch), "batch" (i.e., every
      batch).
    - model_save_freq (int): Frequency (in epochs) for saving models.
  - CL parameters:
    - cl_framework (str): CL framework to use. Valid values are:
      - "finetune": Fine-tuning, using the pretrained model weights in
        `pretrained_dir`. If no `pretrained_dir` is specified, training is
        performed from scratch.
    - pretrained_dir (str): Directory containing the pretrained model weights.
  """
  # Network parameters.
  network_params = {
      'architecture': 'unet',
      'backbone': "vgg16",
      #TODO (fmilano): Retrieve from first training sample.
      'image_h': 480,
      'image_w': 640
  }

  # Training parameters.
  training_params = {
      'batch_size': 8,
      'learning_rate': 1e-5,
      'num_training_epochs': 3
  }

  # Dataset parameters.
  dataset_params = {
      'test_dataset': "NyuDepthV2Labeled",
      'test_scene': None,
      'train_dataset': "BfsegCLAMeshdistLabels",
      'train_scene': None,
      'validation_percentage': 20
  }

  # Logging parameters.
  logging_params = {
      'metric_log_frequency': "batch",
      'model_save_freq': 1,
      'exp_name': "exp_stage1"
  }

  # CL parameters.
  cl_params = {'cl_framework': "finetune", 'pretrained_dir': None}


@ex.main
def run(_run, network_params, training_params, dataset_params, logging_params,
        cl_params):
  r"""Runs the whole training pipeline.
  """
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  print("Current time is " + current_time)
  seg_experiment = BaseSegExperiment(run=_run, root_output_dir=TMPDIR)
  # Set up the experiment.
  seg_experiment.make_dirs()
  seg_experiment.build_model()
  seg_experiment.build_loss_and_metric()
  train_ds, val_ds, test_ds = seg_experiment.load_datasets(
      train_dataset=dataset_params['train_dataset'],
      train_scene=dataset_params['train_scene'],
      test_dataset=dataset_params['test_dataset'],
      test_scene=dataset_params['test_scene'],
      batch_size=training_params['batch_size'],
      validation_percentage=dataset_params['validation_percentage'])
  # Run the training.
  seg_experiment.training(train_ds, val_ds, test_ds)
  # Save final model.
  seg_experiment.save_model()


if __name__ == "__main__":
  ex.run_commandline()
