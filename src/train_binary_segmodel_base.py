import sys
sys.path.append("..")
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import datetime
from sacred import Experiment
from shutil import make_archive

from bfseg.cl_experiments import BaseSegExperiment
from bfseg.sacred_utils import get_observer
from bfseg.settings import TMPDIR

#TODO(fmilano): Pass this as argument to BaseSegExperiment class.
ex = Experiment()
ex.observers.append(get_observer())


@ex.config
def seg_experiment_default_config():
  r"""Default configuration for base segmentation experiments.
  - batch_size (int): Batch size.
  - num_training_epochs (int): Number of training epochs
  - image_w (int): Image width.
  - image_f (int): Image height.
  - validation_percentage (int): Percentage of the training scene to use for
      validation.
  - exp_name (str): Name of the current experiment.
  - backbone (str): Name of the backbone of the U-Net architecture.
  - learning_rate (float): Learning rate.
  - train_dataset (str): Name of the training dataset.
  - test_dataset (str): Name of the test dataset.
  - train_scene (str): Scene type of the training dataset. Valid values are:
      None, "kitchen", "bedroom".
  - test_scene (str): Scene type of the test dataset. Valid values are: None,
      "kitchen", "bedroom".
  - pretrained_dir (str): Directory containing the pretrained model weights.
  - metric_log_frequency (str): Frequency with which the training metrics are
      logged. Valid values are "epoch" (i.e., every epoch), "batch" (i.e., every
      batch).
  - model_save_freq (int): Frequency (in epochs) for saving models.
  """
  batch_size = 8
  num_training_epochs = 3
  #TODO (fmilano): Retrieve from first training sample.
  image_w = 640
  image_h = 480
  validation_percentage = 20

  exp_name = "exp_stage1"
  backbone = "vgg16"
  learning_rate = 1e-5
  train_dataset = "BfsegCLAMeshdistLabels"
  test_dataset = "NyuDepthV2Labeled"
  train_scene = None
  test_scene = None
  pretrained_dir = None
  metric_log_frequency = "batch"
  model_save_freq = 1


@ex.main
def run(_run, batch_size, num_training_epochs, image_w, image_h,
        validation_percentage, exp_name, backbone, learning_rate, train_dataset,
        test_dataset, train_scene, test_scene, pretrained_dir,
        metric_log_frequency, model_save_freq):
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
      train_dataset=train_dataset,
      train_scene=train_scene,
      test_dataset=test_dataset,
      test_scene=test_scene,
      batch_size=batch_size,
      validation_percentage=validation_percentage)
  # Run the training.
  seg_experiment.training(train_ds, val_ds, test_ds)
  # Save the data to sacred.
  path_to_archive_model = make_archive(seg_experiment.model_save_dir, 'zip',
                                       seg_experiment.model_save_dir)
  _run.add_artifact(path_to_archive_model)


if __name__ == "__main__":
  ex.run_commandline()
