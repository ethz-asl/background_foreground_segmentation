import os
import tensorflow as tf
from shutil import make_archive
from tensorflow import keras
import warnings

from bfseg.utils.metrics import get_balanced_weights
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
    # Set up the experiment.
    self._make_dirs()
    self._build_model()
    self._build_loss_and_metric()

  def _make_dirs(self):
    try:
      os.makedirs(self.model_save_dir)
    except os.error:
      pass

  def _build_model(self):
    r"""Builds the models.
    """
    cl_framework = self.run.config['cl_params']['cl_framework']
    assert (cl_framework in [
        "distillation", "ewc", "finetune"
    ]), "Currently, only distillation, EWC and fine-tuning are supported."
    # NOTE: by default the model is created as trainable. The encoder can be
    # optionally be set as non-trainable through the config file.
    # CL frameworks that require a fixed, non-trainable network from which to
    # distill the information (e.g., in distillation experiments) should create
    # additional models by overloading this method and calling
    # `super()._build_model()` in the overload.
    pretrained_dir = self.run.config['cl_params']['pretrained_dir']
    should_freeze_encoder = self.run.config['network_params']['freeze_encoder']
    if (should_freeze_encoder):
      if (cl_framework not in ["distillation", "finetune"]):
        raise ValueError(
            "Freezing the encoder is only done for finetuning. If you are "
            "really sure that you want to do so for other CL frameworks, "
            "manually remove this check.")
      if (pretrained_dir is None):
        raise ValueError(
            "It is necessary to specify pretrained weights to be loaded if the "
            "encoder is to be fixed.")
      warnings.warn(
          "The encoder of the created model is set to be non-trainable.")
    self.encoder, self.model = create_model(
        model_name=self.run.config['network_params']['architecture'],
        freeze_encoder=should_freeze_encoder,
        freeze_whole_model=False,
        normalization_type=self.run.config['network_params']
        ['normalization_type'],
        **self.run.config['network_params']['model_params'])
    self.new_model = keras.Model(
        inputs=self.model.input,
        outputs=[self.encoder.output, self.model.output])
    # Optionally load the model weights.
    if (pretrained_dir is not None):
      print(f"Loading pre-trained weights from {pretrained_dir}.")
      self.new_model.load_weights(pretrained_dir)

  def _build_loss_and_metric(self):
    r"""Adds loss criteria and metrics.
    """
    self.loss_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.loss_tracker = keras.metrics.Mean('loss', dtype=tf.float32)
    self.accuracy_tracker = keras.metrics.Accuracy('accuracy', dtype=tf.float32)
    self.miou_tracker = keras.metrics.MeanIoU(num_classes=2, name='mean_iou')
    # This stores optional logs about the test metrics, which is not
    # automatically handled by Keras.
    self.logs_test = {}
    # By default, no auxiliary losses are expected to be tracked.
    self._tracked_auxiliary_losses = None
    # Check if balanced cross-entropy loss should be used.
    if (self.run.config['training_params']['use_balanced_loss']):
      assert (self.run.config['cl_params']['cl_framework'] == "finetune"
             ), "Currently balanced loss is only supported with finetuning."
      self._use_balanced_loss = True
    else:
      self._use_balanced_loss = False

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
    if (self._use_balanced_loss):
      #TODO(fmilano): Make flexible to number of classes.
      sample_weight = get_balanced_weights(labels=y_masked, num_classes=2)
    else:
      sample_weight = None
    loss = self.loss_ce(y_masked, pred_y_masked, sample_weight=sample_weight)

    return pred_y, pred_y_masked, y_masked, loss

  def _handle_multiple_losses(self, loss):
    r"""Since derived classes might implement auxiliary losses that one also
    wants to keep track of, this method is used to separate the main (total)
    loss from the auxiliary ones. It also sets up loss trackers for the
    auxiliary losses.
    
    Args:
      loss (tensorflow.python.keras.losses.LossFunctionWrapper or dict): If a
        dict, it contains all the losses to consider, indexed by their name; in
        this case, it is assumed that the total loss is named `loss`, and the
        other ones are tracked with their name. If not a dict, the single loss
        to consider.

    Returns:
      total_loss (tensorflow.python.keras.losses.LossFunctionWrapper): Main,
        total loss.
      auxiliary_losses (dict): Dict with the auxiliary losses indexed by their
        name. `None` if no auxiliary losses are present.
    """

    if (isinstance(loss, dict)):
      try:
        total_loss = loss.pop('loss')
        auxiliary_losses = loss
        # Set up trackers for the auxiliary losses if necessary.
        if (self._tracked_auxiliary_losses is None):
          self._tracked_auxiliary_losses = []
          for loss_name in auxiliary_losses:
            setattr(self, f"{loss_name}_tracker",
                    keras.metrics.Mean(f'{loss_name}', dtype=tf.float32))
            self._tracked_auxiliary_losses.append(loss_name)
      except KeyError:
        raise KeyError(
            "When returning more than one loss, there must be a main (total) "
            "loss named `loss`.")
    else:
      total_loss = loss
      auxiliary_losses = None

    return total_loss, auxiliary_losses

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

    total_loss, auxiliary_losses = self._handle_multiple_losses(loss)

    grads = tape.gradient(total_loss, self.new_model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_weights))
    pred_y = tf.math.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, train_mask)

    # Update accuracy and loss.
    self.accuracy_tracker.update_state(train_y_masked, pred_y_masked)
    self.loss_tracker.update_state(total_loss)
    self.miou_tracker.update_state(train_y_masked, pred_y_masked)
    if (auxiliary_losses is not None):
      for aux_loss_name, aux_loss in auxiliary_losses.items():
        getattr(self, f"{aux_loss_name}_tracker").update_state(aux_loss)

    self.current_batch += 1
    self.performed_test_evaluation = False

    return {m.name: m.result() for m in self.metrics}

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

    total_loss, auxiliary_losses = self._handle_multiple_losses(loss)

    # Update val/test metrics.
    self.loss_tracker.update_state(total_loss)
    self.accuracy_tracker.update_state(test_y_masked, pred_y_masked)
    self.miou_tracker.update_state(test_y_masked, pred_y_masked)
    if (auxiliary_losses is not None):
      for aux_loss_name, aux_loss in auxiliary_losses.items():
        getattr(self, f"{aux_loss_name}_tracker").update_state(aux_loss)

    return {m.name: m.result() for m in self.metrics}

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

  def log_metrics(self, metric_type, logs, step):
    r"""Logs to sacred the metrics for the given dataset type ("train",
    "train_no_replay", "val", or "test") at the given step.

    Args:
      metric_type (str): Either "train", "train_no_replay", "val", "test", or
        the name of a dataset on which the model was evaluated: type of the
        dataset the metrics refer to.
      logs (dict): Dictionary containing the metrics to log, indexed by the
        metric name, with their value at the given step.
      step (int): Training epoch to which the metrics are referred.

    Returns:
      None.
    """
    for metric_name, metric_value in logs.items():
      self.run.log_scalar(f'{metric_type}_{metric_name}',
                          metric_value,
                          step=step)
    self.loss_tracker.reset_states()
    self.accuracy_tracker.reset_states()
    self.miou_tracker.reset_states()
    if (self._tracked_auxiliary_losses is not None):
      for aux_loss_name in self._tracked_auxiliary_losses:
        getattr(self, f"{aux_loss_name}_tracker").reset_states()

  def log_lr(self, step):
    r"""Logs the learning rate at the given input step.

    Args:
      step (int): Training epoch at which the learning rate is logged.

    Returns:
      None.
    """
    self.run.log_scalar("lr", self.optimizer.lr.numpy(), step=step)

  @property
  def metrics(self):
    r"""Overrides `tf.keras.Model.metrics.
    """
    auxiliary_losses = []
    if (self._tracked_auxiliary_losses is not None):
      auxiliary_losses = [
          getattr(self, f"{loss_name}_tracker")
          for loss_name in self._tracked_auxiliary_losses
      ]
    return [self.loss_tracker, self.accuracy_tracker, self.miou_tracker
           ] + auxiliary_losses