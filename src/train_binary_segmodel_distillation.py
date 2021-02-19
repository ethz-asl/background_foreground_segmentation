import datetime
from sacred import Experiment
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau

from bfseg.cl_models import DistillationModel
from bfseg.sacred_utils import get_observer
from bfseg.settings import TMPDIR
from bfseg.utils.callbacks import (EarlyStoppingMinimumEpoch, SaveModelAndLogs,
                                   TestCallback)
from bfseg.utils.datasets import (load_datasets, load_replay_datasets,
                                  update_datasets_with_replay_and_augmentation)
from bfseg.utils.images import augmentation
from bfseg.utils.replay_buffer import ReplayBuffer

ex = Experiment()
ex.observers.append(get_observer())


@ex.main
def run(_run, network_params, training_params, dataset_params, logging_params,
        cl_params):
  r"""Runs the whole training pipeline.
  """
  assert (cl_params['cl_framework'] == "distillation"), (
      "The current training script will perform distillation. Please select "
      "CL-framework `distillation`.")
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  print("Current time is " + current_time)
  # Get the datasets.
  train_no_replay_ds, val_ds, test_ds = load_datasets(
      train_dataset=dataset_params['train_dataset'],
      train_scene=dataset_params['train_scene'],
      test_dataset=dataset_params['test_dataset'],
      test_scene=dataset_params['test_scene'],
      batch_size=training_params['batch_size'],
      validation_percentage=dataset_params['validation_percentage'])
  # Instantiate the model.
  model = DistillationModel(run=_run, root_output_dir=TMPDIR)
  # Run the training.
  model.compile(
      optimizer=tf.keras.optimizers.Adam(training_params['learning_rate']))
  # Obtain actual training and test datasets after optionally creating a replay
  # buffer and performing data augmentation.
  train_ds, test_ds = update_datasets_with_replay_and_augmentation(
      train_no_replay_ds=train_no_replay_ds,
      test_ds=test_ds,
      fraction_replay_ds_to_use=cl_params['fraction_replay_ds_to_use'],
      ratio_main_ds_replay_ds=cl_params['ratio_main_ds_replay_ds'],
      replay_datasets=dataset_params['replay_datasets'],
      replay_datasets_scene=dataset_params['replay_datasets_scene'],
      batch_size=training_params['batch_size'],
      perform_data_augmentation=training_params['perform_data_augmentation'])

  model.fit(train_ds,
            epochs=training_params['num_training_epochs'],
            validation_data=val_ds,
            verbose=2,
            callbacks=[
                TestCallback(test_data=test_ds),
                SaveModelAndLogs(),
                ReduceLROnPlateau(),
                EarlyStoppingMinimumEpoch(
                    min_epoch=training_params['stopping_min_epoch'],
                    patience=training_params['stopping_patience'])
            ])
  # Save final model.
  model.save_model(epoch="final")


if __name__ == "__main__":
  ex.run_commandline()
