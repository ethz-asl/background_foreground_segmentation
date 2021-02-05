import os
import tensorflow as tf
from shutil import make_archive
from tensorflow import keras

from bfseg.utils.models import create_model


class BaseCLModel(keras.Model):
  r"""Base class to specify a CL model. It supports the following:
  - Creating a trainable model;
  - Creating loss functions specific for the CL framework;
  - Logging the metrics;
  - Training the model.

  Args:
    run (sacred.run.Run): Object identifying the current sacred run.
    root_output_dir (str): Path to the folder that will contain the experiment
      logs and the saved models.
  """

  def __init__(self, run, root_output_dir):
    super(BaseCLModel, self).__init__()
    self.run = run
    self.model_save_dir = os.path.join(
        root_output_dir, self.run.config['logging_params']['exp_name'],
        'models')
    # Counter for the current training epoch/batch.
    self.current_batch = 0
    # Whether or not test evaluation was performed.
    self.performed_test_evaluation = False
    # Evaluation type (either validation or testing). Since tf.keras.Model.fit
    # first calls validation, and then testing can be optionally run through a
    # custom `on_epoch_end` callback, here the evaluation type is set to
    # validation.
    self.evaluation_type = "val"
    # Logging parameter.
    self.metric_log_frequency = self.run.config['logging_params'][
        'metric_log_frequency']
    assert (self.metric_log_frequency in ["batch", "epoch"])
    # Set up the experiment.
    self._make_dirs()
    self._build_model()
    self._build_loss_and_metric()

  def _make_dirs(self):
    try:
      os.makedirs(self.model_save_dir)
    except os.error:
      pass

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

  def _build_model(self):
    r"""Builds the models.
    TODO(fmilano): Check. Make flexible.
    """
    assert (self.run.config['cl_params']['cl_framework'] == "finetune"
           ), "Currently, only fine-tuning is supported."
    self.encoder, self.model = create_model(
        model_name=self.run.config['network_params']['architecture'],
        **self.run.config['network_params']['model_params'])
    self.new_model = keras.Model(
        inputs=self.model.input,
        outputs=[self.encoder.output, self.model.output])
    # Optionally load the model weights.
    pretrained_dir = self.run.config['cl_params']['pretrained_dir']
    if (pretrained_dir is not None):
      print(f"Loading pre-trained weights from f{pretrained_dir}.")
      self.new_model.load_weights(pretrained_dir)

  def _build_loss_and_metric(self):
    r"""Adds loss criteria and metrics.
    TODO(fmilano): Check. Make flexible.
    """
    self.loss_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.loss_trackers = {
        dataset_type: keras.metrics.Mean(f'{dataset_type}_loss',
                                         dtype=tf.float32)
        for dataset_type in ["test", "train", "val"]
    }
    self.accuracy_trackers = {
        dataset_type: keras.metrics.Mean(f'{dataset_type}_accuracy',
                                         dtype=tf.float32)
        for dataset_type in ["test", "train", "val"]
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
      Dictionary of metrics.
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
    if (self.metric_log_frequency == "batch"):
      self.log_metrics(metric_type='train', step=self.current_batch)

    self.current_batch += 1
    self.performed_test_evaluation = False

    # Only return the training metrics here.
    exclude_metrics = ["test", "val"]

    metrics_to_return = {}

    for m in self.metrics:
      prefix, metric_name = m.name.split("_", maxsplit=1)
      if (prefix in exclude_metrics):
        continue
      # Remove prefix from metrics kept (it is added by `keras.Model.fit()`).
      if (prefix == "train"):
        metrics_to_return[metric_name] = m.result()
      else:
        metrics_to_return[m.name] = m.result()

    return metrics_to_return

  def test_step(self, data):
    r"""Performs one evaluation (test/validation) step with the input batch.
    Overrides `test_step` called internally by the `evaluate` method of the
    `tf.keras.Model`.

    Args:
      data (tuple): Input batch. It is expected to be of the form
        (test_x, test_y, test_mask), where:
        - test_x (tf.Tensor) is the input sample batch.
        - test_y (tf.Tensor) are the ground-truth labels associated to the input
            sample batch.
        - test_mask (tf.Tensor) is a boolean mask for each pixel in the input
            samples. Pixels with `True` mask are considered for the computation
            of the loss.
    
    Returns:
      Dictionary of metrics.
    """
    assert (self.evaluation_type in ["test", "val"])
    test_x, test_y, test_mask = data
    pred_y, pred_y_masked, test_y_masked, loss = self.forward_pass(
        training=False, x=test_x, y=test_y, mask=test_mask)
    pred_y = keras.backend.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, test_mask)
    # Update val/test metrics.
    self.loss_trackers[self.evaluation_type].update_state(loss)
    self.accuracy_trackers[self.evaluation_type].update_state(
        test_y_masked, pred_y_masked)

    # Only return the test/val metrics here, according to the evaluation mode.
    if (self.evaluation_type == "test"):
      exclude_metrics = ["train", "val"]
    else:
      exclude_metrics = ["test", "train"]

    metrics_to_return = {}

    for m in self.metrics:
      prefix, metric_name = m.name.split("_", maxsplit=1)
      if (prefix in exclude_metrics):
        continue
      # Remove prefix from metrics kept (it is added by `keras.Model.fit()`).
      if (prefix == self.evaluation_type):
        metrics_to_return[metric_name] = m.result()
      else:
        metrics_to_return[m.name] = m.result()

    return metrics_to_return

  def save_model(self, epoch):
    r"""Saves the current model both to the local folder and to sacred.

    Args:
      epoch (int or str): Number of the current epoch, or string. In both cases,
        it used to determine the checkpoint filename.

    Returns:
      None.
    """
    model_filename = f'model_epoch_{epoch}'

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
