import os
import tensorflow as tf
from shutil import make_archive
from tensorflow import keras

from bfseg.utils.datasets import load_data
from bfseg.utils.models import create_model


class BaseSegExperiment(keras.Model):
  r"""Base class to specify an experiment. An experiment is a standalone class
  that supports:
  - Loading training data
  - Creating models that can be trained
  - Creating experiment-specific loss functions
  - Logging the metrics
  - Training the model

  Args:
    run (sacred.run.Run): Object identifying the current sacred run.
    root_output_dir (str): Path to the folder that will contain the experiment
      logs and the saved models.
  """

  def __init__(self, run, root_output_dir):
    super(BaseSegExperiment, self).__init__()
    self.run = run
    self.model_save_dir = os.path.join(
        root_output_dir, self.run.config['logging_params']['exp_name'],
        'models')
    self.optimizer = keras.optimizers.Adam(
        self.run.config['training_params']['learning_rate'])
    # Counter for the current training epoch/batch.
    self._current_epoch = None
    self._current_batch = None
    self._completed_training = False
    # Logging parameter.
    self._metric_log_frequency = self.run.config['logging_params'][
        'metric_log_frequency']
    assert (self._metric_log_frequency in ["batch", "epoch"])

  def make_dirs(self):
    try:
      os.makedirs(self.model_save_dir)
    except os.error:
      pass

  def load_datasets(self, train_dataset, train_scene, test_dataset, test_scene,
                    batch_size, validation_percentage):
    r"""Creates 3 data loaders, for training, validation and testing.
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
    assert (self.run.config['cl_params']['cl_framework'] == "finetune"
           ), "Currently, only fine-tuning is supported."
    self.encoder, self.model = create_model(
        model_name=self.run.config['network_params']['architecture'],
        pretrained_dir=self.run.config['cl_params']['pretrained_dir'],
        **self.run.config['network_params']['model_params'])
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
      y_masked (tf.Tensor): Masked ground-truth labels (i.e., with labels only
        from the selected samples).
      loss (tf.Tensor): Loss from performing the forward pass.
    """
    [_, pred_y] = self.new_model(x, training=training)
    pred_y_masked = tf.boolean_mask(pred_y, mask)
    y_masked = tf.boolean_mask(y, mask)
    loss = self.loss_ce(y_masked, pred_y_masked)

    return pred_y, pred_y_masked, y_masked, loss

  def train_step(self, data):
    r"""Performs one training step with the input batch. Overrides `train_step`
    called internally by the `fit` method of the `tf.keras.Model`.

    Args:
      data (tuple): Input batch. It is expected to be of the form
        (train_x, train_y, train_mask), where:
        - train_x (tf.Tensor) is the input sample batch.
        - train_y (tf.Tensor) are the ground-truth labels associated to the
            input sample batch.
        - train_mask (tf.Tensor) is a boolean mask for each pixel in the input
            samples. Pixels with `True` mask are considered for the computation
            of the loss.
    
    Returns:
      None.
    """
    train_x, train_y, train_mask = data
    with tf.GradientTape() as tape:
      pred_y, pred_y_masked, train_y_masked, loss = self.forward_pass(
          training=True, x=train_x, y=train_y, mask=train_mask)
    grads = tape.gradient(loss, self.new_model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_weights))
    pred_y = tf.math.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, train_mask)
    # Update accuracy and loss.
    self.accuracy_trackers['train'].update_state(train_y_masked, pred_y_masked)
    self.loss_trackers['train'].update_state(loss)

    # Log loss and accuracy.
    if (self._metric_log_frequency == "batch"):
      self.log_metrics(metric_type='train', step=self._current_batch)

  def test_step(self, test_x, test_y, test_mask, dataset_type):
    r"""Performs one evaluation (test/validation) step with the input batch.

    Args:
      test_x (tf.Tensor): Input sample batch.
      test_y (tf.Tensor): Ground-truth labels associated to the input sample
        batch.
      test_mask (tf.Tensor): Boolean mask for each pixel in the input samples.
        Pixels with `True` mask are considered for the computation of the loss.
      dataset_type (str): Either "test" or "val".
    
    Returns:
      None.
    """
    assert (dataset_type in ["test", "val"])
    pred_y, pred_y_masked, test_y_masked, loss = self.forward_pass(
        training=False, x=test_x, y=test_y, mask=test_mask)
    pred_y = keras.backend.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, test_mask)
    # Update val/test metrics.
    self.loss_trackers[dataset_type].update_state(loss)
    self.accuracy_trackers[dataset_type].update_state(test_y_masked,
                                                      pred_y_masked)

  def on_epoch_end(self, val_ds, test_ds):
    # Optionally save the model at the end of the current epoch.
    if ((self._current_epoch + 1) %
        self.run.config['logging_params']['model_save_freq'] == 0):
      self.save_model()
    # Optionally log the metrics.
    if (self._metric_log_frequency == "epoch"):
      self.log_metrics(metric_type='train', step=self._current_epoch)
      val_test_logging_step = self._current_epoch
    else:
      val_test_logging_step = self._current_batch
    # Evaluate on validation set.
    for val_x, val_y, val_mask in val_ds:
      self.test_step(val_x, val_y, val_mask, dataset_type="val")
    self.log_metrics("val", step=val_test_logging_step)
    # Evaluate on test set.
    for test_x, test_y, test_mask in test_ds:
      self.test_step(test_x, test_y, test_mask, dataset_type="test")
    self.log_metrics("test", step=val_test_logging_step)

  def save_model(self):
    r"""Saves the current model both to the local folder and to sacred.

    Args:
      None.

    Returns:
      None.
    """
    if (self._completed_training):
      model_filename = 'model_final'
    else:
      model_filename = f'model_epoch_{self._current_epoch}'
    full_model_filename = os.path.join(self.model_save_dir,
                                       f'{model_filename}.h5')
    self.new_model.save(full_model_filename)
    # Save the model to sacred.
    path_to_archive_model = make_archive(
        os.path.join(self.model_save_dir, model_filename), 'zip',
        self.model_save_dir, f'{model_filename}.h5')
    self.run.add_artifact(path_to_archive_model)

  def log_metrics(self, metric_type, step):
    assert (metric_type in ["train", "test", "val"])
    self.run.log_scalar(f'{metric_type}_loss',
                        self.loss_trackers[metric_type].result().numpy(),
                        step=step)
    self.run.log_scalar(f'{metric_type}_accuracy',
                        self.accuracy_trackers[metric_type].result().numpy(),
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
    assert (self._current_epoch is None and self._current_batch is None and
            not self._completed_training
           ), "Currently, only training from epoch 0 is supported."
    self._current_batch = 0
    for self._current_epoch in range(
        self.run.config['training_params']['num_training_epochs']):
      print("\nStart of epoch %d" % (self._current_epoch,))
      for train_sample in train_ds:
        self.train_step(data=train_sample)
        self._current_batch += 1
      self.on_epoch_end(val_ds=val_ds, test_ds=test_ds)
    self._completed_training = True
