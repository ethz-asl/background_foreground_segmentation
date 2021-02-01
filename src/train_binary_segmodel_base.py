import sys
sys.path.append("..")
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import datetime
import tensorflow as tf
from tensorflow import keras
from bfseg.sacred_utils import get_observer
from bfseg.settings import TMPDIR
from bfseg.utils.datasets import load_data
from sacred import Experiment
import segmentation_models as sm
from shutil import make_archive

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


class BaseSegExperiment:
  r"""Base class to specify an experiment. An experiment is a standalone class
  that supports:
  - Loading training data
  - Creating models that can be trained
  - Creating experiment-specific loss functions
  - Logging the metrics
  - Training the model
  """

  def __init__(self, run):
    self.run = run
    self.log_dir = os.path.join(TMPDIR, ex.current_run.config['exp_name'],
                                'logs')
    self.model_save_dir = os.path.join(TMPDIR,
                                       ex.current_run.config['exp_name'],
                                       'models')
    self.optimizer = keras.optimizers.Adam(
        ex.current_run.config['learning_rate'])

  def make_dirs(self):
    try:
      os.makedirs(self.model_save_dir)
    except os.error:
      pass

  def load_datasets(self, train_dataset, train_scene, test_dataset, test_scene,
                    batch_size, validation_percentage):
    r"""Creates 3 data loaders, for training, validation and testing..
    """
    assert (isinstance(validation_percentage, int) and
            0 <= validation_percentage <= 100)
    training_percentage = 100 - validation_percentage
    train_ds = load_data(dataset_name=train_dataset,
                         scene_type=train_scene,
                         fraction=f"[:{training_percentage}%]",
                         batch_size=batch_size,
                         shuffle_data=True)
    val_ds = load_data(dataset_name=train_dataset,
                       scene_type=train_scene,
                       fraction=f"[{training_percentage}%:]",
                       batch_size=batch_size,
                       shuffle_data=False)
    test_ds = load_data(dataset_name=test_dataset,
                        scene_type=test_scene,
                        fraction=None,
                        batch_size=batch_size,
                        shuffle_data=False)
    return train_ds, val_ds, test_ds

  def create_old_params(self):
    r"""Stores the old weights of the model.
    TODO(fmilano): Check.
    """
    pass

  def create_fisher_params(self, dataset):
    r"""Computes squared Fisher information, representing relative importance.
    TODO(fmilano): Check.
    """
    pass

  def compute_consolidation_loss(self):
    r"""Computes weight regularization term.
    TODO(fmilano): Check.
    """
    pass

  def build_model(self):
    r"""Builds the models.
    TODO(fmilano): Check. Make flexible.
    """
    self.encoder, self.model = sm.Unet(
        backbone_name=ex.current_run.config['backbone'],
        input_shape=(ex.current_run.config['image_h'],
                     ex.current_run.config['image_w'], 3),
        classes=2,
        activation='sigmoid',
        weights=ex.current_run.config['pretrained_dir'],
        encoder_freeze=False)
    self.new_model = keras.Model(
        inputs=self.model.input,
        outputs=[self.encoder.output, self.model.output])

  def build_loss_and_metric(self):
    r"""Adds loss criteria and metrics.
    TODO(fmilano): Check. Make flexible.
    """
    self.loss_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.loss_trackers = {
        'train': keras.metrics.Mean('training_loss', dtype=tf.float32),
        'test': keras.metrics.Mean('test_loss', dtype=tf.float32),
        'val': keras.metrics.Mean('validation_loss', dtype=tf.float32)
    }
    self.accuracy_trackers = {
        'train': keras.metrics.Accuracy('training_accuracy'),
        'test': keras.metrics.Accuracy('test_accuracy'),
        'val': keras.metrics.Accuracy('validation_accuracy')
    }

  def forward_pass(self, training, x, y, mask):
    r"""Forward pass.

    Args:
      training (bool): Whether or not the model should be in training mode.
      x (tf.Tensor): Input to the network.
      y (tf.Tensor): Ground-truth labels corresponding to the given input.
      mask (tf.Tensor): Mask for the input to consider.

    Return:
      pred_y (tf.Tensor): Network prediction.
      pred_y_masked (tf.Tensor): Masked network prediction.
      loss (tf.Tensor): Loss from performing the forward pass.
    """
    [_, pred_y] = self.new_model(x, training=training)
    pred_y_masked = tf.boolean_mask(pred_y, mask)
    y_masked = tf.boolean_mask(y, mask)
    loss = self.loss_ce(y_masked, pred_y_masked)

    return pred_y, pred_y_masked, loss

  def train_step(self, train_x, train_y, train_mask, step):
    r"""Performs one training step with the input batch.

    Args:
      train_x (tf.Tensor): Input sample batch.
      train_y (tf.Tensor): Ground-truth labels associated to the input sample
        batch.
      train_mask (tf.Tensor): Boolean mask for each pixel in the input samples.
        Pixels with `True` mask are considered for the computation of the loss.
      step (int): Step number.
    
    Returns:
      None.
    """
    with tf.GradientTape() as tape:
      self.forward_pass(training=True, x=train_x, y=train_y, mask=train_mask)
    grads = tape.gradient(loss, self.new_model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_weights))
    pred_y = tf.math.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, train_mask)
    # Update accuracy and loss.
    self.accuracy_trackers['train'].update_state(train_y_masked, pred_y_masked)
    self.loss_trackers['train'].update_state(loss)

    # Log loss and accuracy.
    metric_log_frequency = ex.current_run.config['metric_log_frequency']
    if (metric_log_frequency == "batch"):
      self.log_metrics(metric_type='train', step=step)

  def test_step(self, test_x, test_y, test_mask, dataset_type):
    r"""Performs one evaluation (test/validation) step with the input batch.

    Args:
      test_x (tf.Tensor): Input sample batch.
      test_y (tf.Tensor): Ground-truth labels associated to the input sample
        batch.
      test_mask (tf.Tensor): Boolean mask for each pixel in the input samples.
        Pixels with `True` mask are considered for the computation of the loss.
      step (int): Step number.
    
    Returns:
      None.
    """
    assert (dataset_type in ["test", "val"])
    self.forward_pass(training=False, x=test_x, y=test_y, mask=test_mask)
    pred_y = keras.backend.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, test_mask)
    # Update val/test metrics.
    self.loss_trackers[dataset_type].update_state(loss)
    self.accuracy_trackers[dataset_type].update_state(test_y_masked,
                                                      pred_y_masked)

  def on_epoch_end(self, epoch, training_step, val_ds, test_ds):
    # Optionally save the model at the end of the current epoch.
    if ((epoch + 1) % ex.current_run.config['model_save_freq'] == 0):
      self.new_model.save(
          os.path.join(self.model_save_dir, f'model_epoch_{epoch}.h5'))
    # Optionally log the metrics.
    metric_log_frequency = ex.current_run.config['metric_log_frequency']
    if (metric_log_frequency == "epoch"):
      self.log_metrics(metric_type='train', step=epoch)
      val_test_logging_step = training_step
    else:
      val_test_logging_step = epoch
    # Evaluate on validation set.
    for val_x, val_y, val_mask in val_ds:
      self.test_step(val_x, val_y, val_mask, dataset_type="val")
    self.log_metrics("val", step=val_test_logging_step)
    # Evaluate on test set.
    for test_x, test_y, test_mask in test_ds:
      self.test_step(test_x, test_y, test_mask, dataset_type="test")
    self.log_metrics("test", step=val_test_logging_step)

  def log_metrics(self, metric_type, step):
    assert (metric_type in ["train", "test", "val"])
    self.run.log_scalar(f'{metric_type}_loss',
                        self.loss_trackers[metric_type].result(),
                        step=step)
    self.run.log_scalar(f'{metric_type}_accuracy',
                        self.accuracy_trackers[metric_type].result(),
                        step=step)
    self.loss_trackers[metric_type].reset_states()
    self.accuracy_trackers[metric_type].reset_states()

  def training(self, train_ds, val_ds, test_ds):
    r"""Performs training for the configured number of epochs. Evaluation on
    validation- and test set is also performed at the end of every epoch. The
    model is saved with the configured epoch frequency.
    
    Args:
      train_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Data
        loader for the training set.
      val_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Data
        loader for the validation set.
      test_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Data
        loader for the test set.
    
    Returns:
      None.
    """
    step = 0
    metric_log_frequency = ex.current_run.config['metric_log_frequency']
    for epoch in range(ex.current_run.config['num_training_epochs']):
      print("\nStart of epoch %d" % (epoch,))
      for train_x, train_y, train_mask in train_ds:
        self.train_step(train_x, train_y, train_mask, step)
        step += 1
      self.on_epoch_end(epoch=epoch,
                        training_step=step,
                        val_ds=val_ds,
                        test_ds=test_ds)


@ex.main
def run(_run, batch_size, num_training_epochs, image_w, image_h,
        validation_percentage, exp_name, backbone, learning_rate, train_dataset,
        test_dataset, train_scene, test_scene, pretrained_dir,
        metric_log_frequency, model_save_freq):
  r"""Runs the whole training pipeline.
  """
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  print("Current time is " + current_time)
  seg_experiment = BaseSegExperiment(run=_run)
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
